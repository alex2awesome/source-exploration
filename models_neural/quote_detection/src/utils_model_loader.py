import logging

from transformers import GPT2Tokenizer, AutoTokenizer, AutoConfig, GPT2LMHeadModel

from . utils_general import reformat_model_path
from . baseline_discriminator import BaselineDiscriminator
from . models_full import Discriminator as SequentialLSTMDiscriminator
# from attrdict import AttrDict
# import torch

# from util.utils_prompting import PromptGenerator


experiments = {
    'baseline': BaselineDiscriminator,
    'lstm_sequential': SequentialLSTMDiscriminator,
    'contextual_flatmap': SequentialLSTMDiscriminator
}

class ModelLoader():
    def __init__(
        self,
        lm_model_type,
        pretrained_model_path,
        pretrained_lm_model_path,
        device,
        #
        discriminator_path,
        experiment,
        # edits model
        perform_edits=False,
        spacy_model=None,
        *args,
        **kwargs
    ):
        self.lm_model_type = lm_model_type
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_lm_model_path = pretrained_lm_model_path
        self.max_num_word_positions = kwargs.get('max_num_word_positions', 2048)
        self.discriminator_path = discriminator_path
        self.device = device
        self.env = kwargs.get('env', 'bb')
        self.spacy_model = spacy_model
        # cache globally in this file, too.
        global _spacy_nlp
        if _spacy_nlp is None and spacy_model is not None:
            _spacy_nlp = spacy_model

        assert experiment in experiments, 'choose one of {"baseline", "lstm_sequential"}'
        self.experiment = experiment

        # load models
        self.perform_edits = perform_edits
        self.edit_discriminator_path = kwargs.get('edit_discriminator_path')
        self.edit_discriminator_config_path = kwargs.get('edit_discrim_config_path')
        self.kwargs = kwargs
        self.config = kwargs.pop('config', None) or kwargs.pop('discrim_config', None)
        self.load_models(**kwargs)
        if self.perform_edits:
            self.load_edit_finder()

    def load_models(self, *args, **kwargs):
        # load pretrained model
        self.pretrained_model_path = reformat_model_path(self.pretrained_model_path)

        # load tokenizer
        if self.lm_model_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model_path)
        elif self.lm_model_type in ["bert", "roberta"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path, output_hidden_states=True)

        if self.discriminator_path is not None:
            self.load_discriminator(*args, **kwargs)

        if self.pretrained_lm_model_path is not None and self.pretrained_lm_model_path.endswith('.ckpt'):
            self.load_finetuned_lm(*args, **kwargs)

    def load_finetuned_lm(self, *args, **kwargs):
        ######### if loading from a checkpoint
        # Initialize the model structure - pytorch_lightning will call `load_state_dict()`.
        # This is lighter-weight than loading the pretrained model just to overwrite the weights.
        transformer_config = AutoConfig.from_pretrained(self.pretrained_model_path)
        transformer_config.vocab_size = self.tokenizer.vocab_size + self.prompt_gen.num_added_tokens
        transformer_config.n_ctx = transformer_config.n_positions = self.max_num_word_positions
        from .language_models import LMModel
        self.lm = LMModel.load_from_checkpoint(
            self.pretrained_lm_model_path,
            map_location=self.device,
            loading_from_checkpoint=True,
            config=transformer_config
        )
        logging.info('loading custom LM onto device: %s ...' % self.device)
        self.lm = self.lm.to(self.device)
        self.config = transformer_config
        self.config.use_headline = True
        self.config.generate = True
        self.config.doc_sent_length_cutoff = kwargs.get('doc_sent_length_cutoff')
        self.tokenizer = self.prompt_gen.resize_tokenizer(self.tokenizer)

    def load_discriminator(self, *args, **kwargs):
        logging.info('loading models onto device: %s...' % self.device)
        # load model and split
        lightning_module = experiments[self.experiment]
        full_discriminator = (
            lightning_module.load_from_checkpoint(
                checkpoint_path=self.discriminator_path,
                loading_from_checkpoint=True,
                pretrained_cache_dir=self.pretrained_model_path,
                config=self.config,
                **kwargs
                # gpt2-medium-expanded-embeddings
            )
        )
        logging.info('loaded models!!')
        self.full_discriminator = full_discriminator.to(self.device)

        # Freeze all weights
        self.full_discriminator.eval()
        for param in self.full_discriminator.parameters():
            param.requires_grad = False
        self.lm = full_discriminator.transformer.to(self.device)
        for param in self.lm.parameters():
            param.requires_grad = False
        self.classifier = full_discriminator.head.to(self.device)
        for param in self.classifier.parameters():
            param.requires_grad = False

_spacy_nlp = None
GENERIC_SPACY_MODEL_NAME = 'en_core'

def get_spacy_nlp(model_name=GENERIC_SPACY_MODEL_NAME):
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load(model_name)
    return _spacy_nlp
