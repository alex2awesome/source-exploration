import transformers
from packaging import version
from torch import nn as nn
from transformers import AutoConfig, GPT2Model, RobertaModel

from models_neural.src.utils_general import get_config


class FFContextMixin(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)
        self.embedding_to_hidden = nn.Linear(self.config.embedding_dim, self.config.hidden_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding_to_hidden.state_dict()['weight'])

    def get_contextualized_embeddings(self, cls_embeddings, *args, **kwargs):
        hidden_output = self.embedding_to_hidden(cls_embeddings)
        return hidden_output

    def get_final_hidden_layer_size(self):
        return self.config.hidden_dim


class BiLSTMContextMixin(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        assert hasattr(self.config, 'num_contextual_layers') and hasattr(self.config, 'bidirectional')
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_contextual_layers,
            bidirectional=self.config.bidirectional
        )
        self._init_lstm_hidden_layers()

    def _init_lstm_hidden_layers(self):
        for name, param in self.lstm.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def get_final_hidden_layer_size(self):
        if self.config.bidirectional:
            return self.config.hidden_dim * 2
        else:
            return self.config.hidden_dim

    def get_contextualized_embeddings(self, cls_embeddings, input_len_eq_one=False, *args, **kwargs):
        if len(cls_embeddings.shape) == 1:
            cls_embeddings = cls_embeddings.unsqueeze(dim=0)
        if cls_embeddings.shape[0] != 1 and len(cls_embeddings.shape) == 2 and input_len_eq_one:
            # what this means is a large batch of inputs has been provided. (IG)
            # and that they were all sequences of length 1.
            # `input_embeds` expects argument of size (batch_size, sequence_length, hidden_size), so we'll unsqueeze
            # so that sequence length is of length 1
            cls_embeddings = cls_embeddings.unsqueeze(dim=1)

        seq_len, embed_size = cls_embeddings.shape
        cls_embeddings = cls_embeddings.reshape(seq_len, 1, embed_size)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(cls_embeddings)
        return lstm_out.reshape(seq_len, self.get_final_hidden_layer_size())


class TransformerContextMixin(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        transformer_config = AutoConfig.from_pretrained(self.config.pretrained_model_path)
        orig_embed_size = transformer_config.hidden_size
        assert hasattr(self.config, 'num_sent_attn_heads') and hasattr(self.config, 'num_contextual_layers')
        super().__init__(*args, **kwargs)
        # resize
        if orig_embed_size != self.config.hidden_dim:
            self.resize_layer = nn.Linear(orig_embed_size, self.config.hidden_dim, bias=False)
            self.do_resize = True
        else:
            self.do_resize = False
        # load transformer
        if self.config.sentence_contextualizer_model_type == 'gpt2':
            transformer_config.n_head = self.config.num_sent_attn_heads
            transformer_config.n_layer = self.config.num_contextual_layers
            transformer_config.n_embd = self.config.hidden_dim
            transformer_config.n_positions = self.config.max_num_sentences + 20
            transformer_config.n_ctx = transformer_config.n_positions
            self.sentence_transformer = GPT2Model(config=transformer_config)
        elif self.config.sentence_contextualizer_model_type == 'roberta':
            transformer_config.num_attention_heads = self.config.num_sent_attn_heads
            transformer_config.num_hidden_layers = self.config.num_contextual_layers
            transformer_config.hidden_size = self.config.hidden_dim
            transformer_config.max_position_embeddings = self.config.max_num_sentences + 20
            self.sentence_transformer = RobertaModel(config=transformer_config)
        else:
            raise NotImplementedError

    def get_final_hidden_layer_size(self):
        return self.config.hidden_dim

    def get_contextualized_embeddings(self, cls_embeddings, input_len_eq_one=None, *args, **kwargs):
        if self.do_resize: # pass vector through a linear layer to resize it
            cls_embeddings = self.resize_layer(cls_embeddings)
        #  a single sentence/doc has been passed in, but flattened.
        if len(cls_embeddings.shape) == 1:
            cls_embeddings = cls_embeddings.unsqueeze(dim=0)

        # inputs_embeds: input of shape: (batch_size, sequence_length, hidden_size)
        contextualized_embeds, _ = self.sentence_transformer(inputs_embeds=cls_embeddings, return_dict=False)
        return contextualized_embeds