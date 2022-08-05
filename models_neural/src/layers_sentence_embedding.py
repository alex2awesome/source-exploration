import torch
import transformers
from packaging import version
from torch import nn as nn
from transformers import AutoConfig, GPT2LMHeadModel, BertModel, RobertaModel

from models_neural.src.layers_attention import WordLevelAttention
from models_neural.src.utils_general import get_config, freeze_all_params, format_layer_freezes, \
    reshape_and_pad_sequence

EPSILON = 1e-10

class PretrainedModelLoader(nn.Module):
    def __init__(self, config, loading_from_checkpoint=False, *args, **kwargs):
        self.config = get_config(config=config, kwargs=kwargs)
        # setup configs
        self.loading_from_checkpoint = loading_from_checkpoint
        if self.loading_from_checkpoint:
            if kwargs.get('pretrained_model_path') != self.config.pretrained_model_path:
                self.config.pretrained_model_path = kwargs.get('pretrained_model_path')

        super().__init__()
        # get encoder
        self.get_pretrained_model()

        # freeze layers
        self.freeze_encoder_layers()

    def get_pretrained_model(self):
        # get pretrained model
        if self.config.model_type == "gpt2":
            transformer_config = AutoConfig.from_pretrained(self.config.pretrained_model_path)
            transformer_config.n_ctx = transformer_config.n_positions = self.config.max_num_word_positions
            self.embed_size = transformer_config.hidden_size
            ######### if loading from a checkpoint
            # Initialize the model structure - pytorch_lightning will call `load_state_dict()`.
            # This is lighter-weight than loading the pretrained model just to overwrite the weights.
            if self.loading_from_checkpoint:
                self.encoder_model = GPT2LMHeadModel(config=transformer_config)
            else:
                self.encoder_model = GPT2LMHeadModel.from_pretrained(self.config.pretrained_model_path, config=transformer_config)
            ##
        elif self.config.model_type == "bert":
            self.encoder_model = BertModel.from_pretrained(self.config.pretrained_model_path)
            self.embed_size = self.encoder_model.config.hidden_size
        elif self.config.model_type == 'roberta':
            self.encoder_model = RobertaModel.from_pretrained(self.config.pretrained_model_path)
            self.embed_size = self.encoder_model.config.hidden_size
        else:
            raise ValueError(
                "{} model not yet supported".format(self.config.model_type)
            )

    def freeze_encoder_layers(self):
        # freeze whole transformer
        if self.config.freeze_transformer:
            freeze_all_params(self.encoder_model)

        # freeze embedding layer
        if self.config.freeze_embedding_layer:
            if self.config.model_type == 'gpt2':
                freeze_all_params(self.encoder_model.transformer.wte)
            else:
                freeze_all_params(self.encoder_model.embeddings)

        # freeze encoding layers
        if self.config.freeze_encoder_layers:
            layers_to_freeze = format_layer_freezes(self.config.freeze_encoder_layers)
            for layer in layers_to_freeze:
                if self.config.model_type == 'gpt2':
                    freeze_all_params(self.encoder_model.transformer.h[layer])
                else:
                    freeze_all_params(self.encoder_model.encoder.layer[layer])


class SentenceEmbeddingsLayer(PretrainedModelLoader):
    """
    Main Discourse generator_pplm class:

        Base Document encoder: RoBERTa
        Head: Bi-LSTM or CRF
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.sentence_embedding_method == 'attention':
            self.additive_attention = WordLevelAttention(config=self.config, input_dim=self.embed_size)

    def get_sentence_embedding(
            self,
            input_ids,
            attention_mask,
            sequence_lens=None,
            inputs_embeds=None,
            get_last=False,
            get_word_embds=False
    ):
        """
        Helper method to calculate sentence embeddings for text.

        Parameters:
            * input_ids: normally, this is a matrix of size (len doc X max len sents) (unless sequence_lens is passed).
            * attention_mask: matrix of size (len doc X max len sents) with zeros to represent input_id padding.
            * sequence_lens: if passed, assume that input_ids is of shape (num docs X total doc len).
            * get_last: if true, return the sentence embedding of the last sentence in the doc.
            * get_word_embs: if true, return the sequence of hidden states.
        """
        hidden = self._get_word_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask if sequence_lens is None else None,
            inputs_embeds=inputs_embeds
        )
        if get_word_embds:
            return hidden

        if sequence_lens is not None:
            if not get_last:
                if len(hidden.shape) == 3: # why do we do this?
                    hidden = list(map(lambda x: reshape_and_pad_sequence(x, sequence_lens), hidden))
                else:
                    hidden = reshape_and_pad_sequence(hidden, sequence_lens)
            else:
                start_of_curr_seq = sequence_lens[-1]
                hidden = hidden[:, -start_of_curr_seq:]

        # aggregate
        if self.config.sentence_embedding_method == 'average':
            return self._avg_representation(hidden, attention_mask)
        elif self.config.sentence_embedding_method == 'cls':
            return self._cls_token(hidden, attention_mask)
        elif self.config.sentence_embedding_method == 'attention':
            return self._attention_representation(hidden, attention_mask)
        else:
            raise NotImplementedError(
                'SENTENCE EMBEDDING METHOD %s not in {average, cls, attention}' % self.config.sentence_embedding_method
            )

    def _get_word_embeddings(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if hasattr(self.encoder_model, 'transformer'):
            # for gpt2
            if version.parse(transformers.__version__) > version.parse('4.0.0'):
                hidden, _ = self.encoder_model.transformer(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
            else:
                hidden, _ = self.encoder_model.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_embeds=inputs_embeds
                )
        else:
            # for bert
            if version.parse(transformers.__version__) > version.parse('4.0.0'):
                hidden, _ = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            else:
                hidden, _ = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        return hidden

    def _avg_representation(self, hidden, attention_mask):
        additive_mask = (
            attention_mask
              .unsqueeze(2)
              .repeat(1, 1, self.embed_size)
              .float()
              .detach()
        )
        masked_hidden = hidden * additive_mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(additive_mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def _cls_token(self, hidden, attention_mask):
        if self.config.model_type == 'roberta':
            cls_embeddings = hidden[:, 0, :]
        elif self.config.model_type == 'gpt2':
            seq_lengths = attention_mask.sum(-1).long() - 1
            cls_embeddings = hidden[range(seq_lengths.size().numel()), seq_lengths, :]  # get hidden states of last input_ids in sequence
        return cls_embeddings

    def _attention_representation(self, hidden, attention_mask):
        if isinstance(hidden, list):
            return list(map(lambda x: self.additive_attention(x, attention_mask), hidden))
        else:
            return self.additive_attention(hidden, attention_mask)

    def get_lmhead_logits_and_past_and_hidden(self, input_ids=None, attention_mask=None, past_key_values=None, input_embeds=None):
        """Pass-through method, here for convenience (Used in the generator_pplm.)"""
        assert (input_ids is not None) or (input_embeds is not None)
        if version.parse(transformers.__version__) > version.parse('4.0.0'):
            # fix input
            if past_key_values is not None:
                if not isinstance(past_key_values[0], tuple):
                    past_key_values = tuple(list(map(lambda l: (l[[0]], l[[1]]), past_key_values)))

            logits, past, all_hidden = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask,
                                                  past_key_values=past_key_values, inputs_embeds=input_embeds,
                                                  use_cache=True, output_hidden_states=True, return_dict=False)
            # Huffingface outputted `past` is of size:
            # if version 3.0.2: (2, batch_size, num_heads, sequence_length, embed_size_per_head)
            # if version >4.0.0: ...
            #   num_layers tuples with:  2, batch_size, num_heads, sequence_length

            # versioning shift
            past = tuple(list(map(lambda l: torch.cat(l), past)))
            return logits, past, all_hidden

        else:
            # version 3.0.2
            return self.encoder_model(input_ids=input_ids, attention_mask=attention_mask,
                                                  past=past_key_values, inputs_embeds=input_embeds,
                                                  use_cache=True, output_hidden_states=True)

    def resize_token_embeddings(self, new_num_tokens=None):
        return self.encoder_model.transformer.resize_token_embeddings(new_num_tokens=new_num_tokens)