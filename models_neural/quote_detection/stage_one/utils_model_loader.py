import logging

from transformers import GPT2Tokenizer, AutoTokenizer, AutoConfig, GPT2LMHeadModel

from generator.utils_general import reformat_model_path
from discriminator.baseline_discriminator import BaselineDiscriminator
from discriminator.models_full import Discriminator as SequentialLSTMDiscriminator
from allennlp.nn.util import move_to_device
from editing.src.stage_two import load_models as load_edit_models
from editing.src.edit_finder import EditFinder, EditEvaluator
from attrdict import AttrDict
import torch
from util.utils_prompting import PromptGenerator


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
        self.prompt_gen = PromptGenerator()
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
        from fine_tuning.language_models import LMModel
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

    def get_editor_config(self):
        editor_config = AttrDict()
        editor_config.pretrained_lm_model_path = self.pretrained_model_path
        editor_config.pretrained_editor_lm_path = self.kwargs.get('editor_pretrained_editor_lm_path')
        editor_config.finetuned_editor_lm_path = self.kwargs.get('editor_finetuned_editor_lm_path')
        editor_config.model_max_length = self.kwargs.get('editor_model_max_length')
        editor_config.grad_type = self.kwargs.get('editor_grad_type')
        editor_config.grad_pred = self.kwargs.get('editor_grad_pred')
        editor_config.search_max_mask_frac = self.kwargs.get('editor_search_max_mask_frac')
        editor_config.search_beam_width = self.kwargs.get('editor_search_beam_width')
        editor_config.search_search_method = self.kwargs.get('editor_search_search_method')
        editor_config.search_max_search_levels = self.kwargs.get('editor_search_max_search_levels')
        editor_config.use_heuristic_masks = self.kwargs.get('editor_use_heuristic_masks')
        editor_config.no_repeat_ngram_size = self.kwargs.get('editor_no_repeat_ngram_size')
        editor_config.generate_type = self.kwargs.get('editor_generate_type')
        editor_config.generation_top_p = self.kwargs.get('editor_generation_top_p')
        editor_config.generation_top_k = self.kwargs.get('editor_generation_top_k')
        editor_config.generation_length_penalty = self.kwargs.get('editor_generation_length_penalty')
        editor_config.num_generations = self.kwargs.get('editor_num_generations')
        editor_config.generation_num_beams = self.kwargs.get('editor_generation_num_beams')
        editor_config.spacy_model = self.spacy_model
        editor_config.spacy_model_file = str(self.spacy_model.path)
        editor_config.local = self.kwargs['env'] == 'local'
        editor_config.real_data_file = None
        return editor_config

    def load_edit_finder(self):
        # editing
        edit_config = self.get_editor_config()
        if self.edit_discriminator_path is not None and self.edit_discriminator_config_path is not None:
            lightning_module = experiments[self.experiment]
            from discriminator.config_helper import TransformersConfig
            import glob
            edit_discrim_config = TransformersConfig.from_json_file(self.edit_discriminator_config_path)
            self.edit_discriminator_path = reformat_model_path(self.edit_discriminator_path)
            print('Edit discrim path: %s' % str(glob.glob(self.edit_discriminator_path)))
            print('Edit config path: %s' % str(glob.glob(self.edit_discriminator_config_path)))
            edit_discriminator = (
                lightning_module.load_from_checkpoint(
                    checkpoint_path=self.edit_discriminator_path,
                    loading_from_checkpoint=True,
                    pretrained_cache_dir=self.pretrained_model_path,
                    config=edit_discrim_config,
                    # gpt2-medium-expanded-embeddings
                )
            )
        else:
            edit_discriminator = self.full_discriminator

        editor, predictor = load_edit_models(args=edit_config, preloaded_hf_predictor=edit_discriminator)
        editor = move_to_device(editor, torch.device(self.device))
        predictor = move_to_device(predictor, torch.device(self.device))
        edit_evaluator = EditEvaluator(
            args=edit_config,
            fluency_model_name=edit_config.pretrained_editor_lm_path,
            spacy_dir=edit_config.spacy_model_file
        )
        self.edit_finder = EditFinder(
            predictor,
            editor,
            edit_evaluator=edit_evaluator,
            beam_width=edit_config.search_beam_width,
            max_mask_frac=edit_config.search_max_mask_frac,
            search_method=edit_config.search_search_method,
            max_search_levels=edit_config.search_max_search_levels,
            success_criteria="increase prob"
        )
        self.edit_finder = move_to_device(self.edit_finder, torch.device(self.device))


_spacy_nlp = None
GENERIC_SPACY_MODEL_NAME = 'en_core'

def get_spacy_nlp(model_name=GENERIC_SPACY_MODEL_NAME):
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load(model_name)
    return _spacy_nlp
