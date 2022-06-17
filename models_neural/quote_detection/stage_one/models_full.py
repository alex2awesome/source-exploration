import torch.nn as nn
from transformers import AutoConfig, RobertaModel, BertModel, GPT2LMHeadModel
import transformers
from packaging import version
if version.parse(transformers.__version__) == version.parse('3.0.2'):
    from transformers.modeling_gpt2 import GPT2Model
else: # transformers: version 4.0
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import torch
from operator import mul
import re
# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)

from .layers_classification import MultiClassMixin, MultiTaskMultiClassMixin
from .layers_embeddings import EmbeddingHandlerMixin
from .layers_attention import (
    WordLevelAttention, DocEmbeddingForDocLabelClass, LabelEmbeddingWithContext
)
from .utils_lightning import LightningMixin
from .utils_general import (
    format_layer_freezes, freeze_all_params, get_config, reshape_and_pad_sequence, vec_or_nones, get_device
)

EPSILON = 1e-10


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
        transformer_config = AutoConfig.from_pretrained(self.config.pretrained_cache_dir)
        orig_embed_size = transformer_config.hidden_size

        # if orig_embed_size != self.config.hidden_dim:
        #     self.resize_layer = nn.Linear(orig_embed_size, self.config.hidden_dim, bias=False)
        #     self.do_resize = True


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
        transformer_config = AutoConfig.from_pretrained(self.config.pretrained_cache_dir)
        orig_embed_size = transformer_config.hidden_size
        assert hasattr(self.config, 'num_sent_attn_heads') and hasattr(self.config, 'num_contextual_layers')
        super().__init__(*args, **kwargs)
        transformer_config.n_head = self.config.num_sent_attn_heads
        transformer_config.n_layer = self.config.num_contextual_layers
        transformer_config.n_embd = self.config.hidden_dim
        transformer_config.n_positions = self.config.max_num_sentences + 20
        transformer_config.n_ctx = transformer_config.n_positions
        # resize
        if orig_embed_size != self.config.hidden_dim:
            self.resize_layer = nn.Linear(orig_embed_size, self.config.hidden_dim, bias=False)
            self.do_resize = True
        else:
            self.do_resize = False
        # load transformer
        if self.config.model_type == 'gpt2':
            self.sentence_transformer = GPT2Model(config=transformer_config)
        else:
            raise NotImplementedError

    def get_final_hidden_layer_size(self):
        return self.config.hidden_dim

    def get_contextualized_embeddings(self, cls_embeddings, input_len_eq_one=None, *args, **kwargs):
        if self.do_resize: # pass vector through a linear layer to resize it
            cls_embeddings = self.resize_layer(cls_embeddings)
        #
        if version.parse(transformers.__version__) > version.parse('4.0.0'):
            #  a single sentence/doc has been passed in, but flattened.
            if len(cls_embeddings.shape) == 1:
                cls_embeddings = cls_embeddings.unsqueeze(dim=0)
            if cls_embeddings.shape[0] != 1 and len(cls_embeddings.shape) == 2 and input_len_eq_one:
                # what this means is a large batch of inputs has been provided. (IG)
                # and that they were all sequences of length 1.
                # `input_embeds` expects argument of size (batch_size, sequence_length, hidden_size), so we'll unsqueeze
                # so that sequence length is of length 1
                cls_embeddings = cls_embeddings.unsqueeze(dim=1)

            # inputs_embeds: input of shape: (batch_size, sequence_length, hidden_size)
            contextualized_embeds, _ = self.sentence_transformer(inputs_embeds=cls_embeddings, return_dict=False)
        else:
            contextualized_embeds, _ = self.sentence_transformer(inputs_embeds=cls_embeddings)
        return contextualized_embeds


class SentenceEmbeddingsLayer(nn.Module):
    """
    Main Discourse generator_pplm class:

        Base Document encoder: RoBERTa
        Head: Bi-LSTM or CRF
    """
    def __init__(self, config=None, loading_from_checkpoint=False, *args, **kwargs):
        # setup configs
        self.config = get_config(config=config, kwargs=kwargs)
        self.loading_from_checkpoint = loading_from_checkpoint
        if self.loading_from_checkpoint:
            if kwargs.get('pretrained_cache_dir') != self.config.pretrained_cache_dir:
                self.config.pretrained_cache_dir = kwargs.get('pretrained_cache_dir')

        super().__init__()

        # get encoder
        self.get_pretrained_model()

        # freeze layers
        self.freeze_encoder_layers()

        # setup dropout
        self.dropout = nn.Dropout(self.config.dropout)

        #
        if self.config.sentence_embedding_method == 'attention':
            self.additive_attention = WordLevelAttention(config=self.config, input_dim=self.embed_size)

    def get_pretrained_model(self):
        # get pretrained model
        if self.config.model_type == "gpt2":
            transformer_config = AutoConfig.from_pretrained(self.config.pretrained_cache_dir)
            transformer_config.n_ctx = transformer_config.n_positions = self.config.max_num_word_positions
            self.embed_size = transformer_config.hidden_size
            ######### if loading from a checkpoint
            # Initialize the model structure - pytorch_lightning will call `load_state_dict()`.
            # This is lighter-weight than loading the pretrained model just to overwrite the weights.
            if self.loading_from_checkpoint:
                self.encoder_model = GPT2LMHeadModel(config=transformer_config)
            else:
                self.encoder_model = GPT2LMHeadModel.from_pretrained(self.config.pretrained_cache_dir, config=transformer_config)
            ##
        elif self.config.model_type == "bert":
            self.encoder_model = BertModel.from_pretrained(self.config.pretrained_cache_dir)
            self.embed_size = self.encoder_model.config.hidden_size
        elif self.config.model_type == 'roberta':
            self.encoder_model = RobertaModel.from_pretrained(self.config.pretrained_cache_dir)
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

    def get_sentence_embedding(self, input_ids, attention_mask, sequence_lens=None,
                               inputs_embeds=None,
                               get_last=False, get_word_embds=False
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

class LabelEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = get_device()
        self.label_embeds = nn.Embedding(self.config.num_output_tags + 2, self.config.hidden_dim)
        if self.config.label_pos_embs:
            self.label_pos_embeds = nn.Embedding(self.config.max_position_embeddings + 1, self.config.hidden_dim)
        self.start_idx = self.config.num_output_tags
        self.end_idx = self.config.num_output_tags + 1
        self.label_attention = LabelEmbeddingWithContext(config=self.config)
        self.s0 = 0 if not self.config.use_headline else 1
        self.s1 = self.s0 + 1

    def get_s_and_e(self):
        if (self.config.num_labels_pred_window == 0
            or (self.config.num_labels_pred_window is None)
            or self.config.separate_heads
        ):
            return 1, 1
        else:
            return 0, 0

    def get_label_idx_range(self, n, head=None):
        if head == 'generate':
            return range(self.s0, n)

        if not self.config.separate_heads:
            if ((self.config.num_labels_pred_window == 0) or (self.config.num_labels_pred_window is None)):
                return range(self.s1, n - 1)
            else:
                return range(self.s0, n)
        else:
            if head == 'main':
                return range(self.s1, n - 1)
            else:
                k = int(re.search('\d', head)[0])
                if 'forward' in head:
                    return list(range(self.s0 + k + 1, n)) + [n - 1] * (k - 1)
                else:
                    return [self.s0] * k + list(range(self.s1, n - k - 1))

    def reformat_labels(self, labels, head=None):
        """labels is padded with (l_s, l_e): i.e.: [l_s, l_0, l_1..., l_n, l_e]"""
        if head == 'generate':
            return labels[self.s0: ]
        if not self.config.separate_heads or head == 'main':
            return labels[self.s1: -1]
        else:
            k = int(re.search('\d', head)[0])
            n = len(labels)
            ignore_tensor = torch.tensor([-100] * (k - 1), device=self.device)
            if 'forward' in head:
                idxs = list(range(self.s0 + k + 1, n))
                output = torch.hstack((labels[idxs], ignore_tensor))
            else:
                idxs = list(range(self.s0, n - k - 1))
                output = torch.hstack((ignore_tensor, labels[idxs]))
            return output.to(int)

    def get_label_embeddings(self, labels, head=None):
        if isinstance(labels, list):
            labels = [self.start_idx] + labels + [self.end_idx]
            labels = torch.tensor(labels, device=self.device)
        else:
            start_lab = torch.tensor([self.start_idx], device=self.device)
            end_lab = torch.tensor([self.end_idx], device=self.device)
            labels = torch.hstack((start_lab, labels, end_lab))

        label_embedding_mat = self.label_embeds(labels)
        if self.config.label_pos_embs:
            position_ids = torch.arange(len(labels), dtype=torch.long, device=self.device)
            position_ids = position_ids.where(position_ids < self.config.max_position_embeddings, torch.tensor(self.config.max_position_embeddings, device=self.device))
            pos_emb = self.label_pos_embeds(position_ids)
            label_embedding_mat = label_embedding_mat + pos_emb

        output_label_embs = []
        to_iterate = self.get_label_idx_range(len(label_embedding_mat), head)
        for label_idx in to_iterate:
            windowed_embedding = self.label_attention(label_embedding_mat, label_idx)
            output_label_embs.append(windowed_embedding)
        output_label_embs = torch.vstack(output_label_embs)
        labels = self.reformat_labels(labels, head)
        return output_label_embs, labels


class HeadBase(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)
        if self.config.do_doc_pred:
            self.doc_embeddings = DocEmbeddingForDocLabelClass(*args, **kwargs)
        if self.config.do_version:
            self.version_emb = nn.Embedding(40, self.config.hidden_dim)
            self.version_ff = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        if not self.config.share_label_embeds:
            if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
                self.label_embds = LabelEmbeddings(config=self.config)

    def handle_x_embeddings(self, sent_embs, add_features=None, get_last=None, generate=False, *args, **kwargs):
        sent_embs = self.get_contextualized_embeddings(sent_embs, kwargs.get('input_len_eq_one'))
        augmented_embs = self.transform_sentence_embeddings(sent_embs)
        if self.config.use_headline and not generate:
            augmented_embs = augmented_embs[1:]
        if self.config.do_doc_pred:
            augmented_embs = self.doc_embeddings(augmented_embs)
        if self.config.do_version:
            augmented_embs = torch.hstack((augmented_embs, self.version_emb(add_features)))
            augmented_embs = self.version_ff(augmented_embs)
        # only used when we're assessing candidates
        if get_last:
            augmented_embs = augmented_embs[-1]
        return augmented_embs

    def handle_y_embeddings(self, labels, label_embs=None, label_idx=None, add_features=None):
        if not self.config.share_label_embeds and self.config.use_y:
            if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
                label_embs, labels = self.label_embds.get_label_embeddings(labels, head=add_features)
        if label_idx is not None and self.config.use_y:
            offset = 0 if not self.config.use_headline else 1
            label_embs = label_embs[label_idx - offset]
        return label_embs, labels

    def forward(self, sent_embs, labels=None, get_last=False,
                add_features=None, label_embs=None, label_idx=None,
                get_loss=True, generate=False, *args, **kwargs):
        """
        Parameters:
            * `sent_embs`: list of sentence embeddings.
            * `labels`: list of labels.

        If labels provided, returns (loss, prediction). Else, returns (None, prediction).
        """
        label_embs, labels = self.handle_y_embeddings(labels, label_embs, label_idx, add_features)
        augmented_embs = self.handle_x_embeddings(sent_embs, add_features, get_last, generate, *args, **kwargs)
        if generate:
            augmented_embs = augmented_embs[[-1]]
            if self.config.use_y:
                offset = 0 if not self.config.use_headline else 1
                labels = labels[[label_idx - offset]]
        if get_loss:
            return self.classification(augmented_embs, labels, label_embs)
        else:
            return self.classification(augmented_embs, None, label_embs)


class HeadLayerFF(MultiClassMixin, FFContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HeadLayerLSTM(MultiClassMixin, BiLSTMContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HeadLayerTransformer(MultiClassMixin, TransformerContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class HeadLayerMultitaskFF(MultiTaskMultiClassMixin, FFContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HeadLayerMultitaskLSTM(MultiTaskMultiClassMixin, BiLSTMContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HeadLayerMultitaskTransformer(MultiTaskMultiClassMixin, TransformerContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BaseDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def is_multitask(self):
        if self.config.separate_heads:
            return False

        return self.config.do_multitask or \
               (self.config.num_labels_pred_window is not None and
               self.config.num_labels_pred_window != 0)

    def forward(self, input_ids=None, add_features=None, labels=None, attention_mask=None,
                input_lens=None, return_lls=False, inputs_embeds=None, *args, **kwargs):
        """
        Step that's shared between training loop and validation loop. Contains sequence-specific processing,
        so we're keeping it in the child class.

        Parameters:
            * `input_ids`: list of docs, len(input_ids) = # batches (default = 1).
                Each item is a flat list of token-ids of length `num_toks_in_doc`.
            * `labels`: [optional] list of sentence-level labels of length batch_size.
                Each item contains tensor of labels length `num_sents_in_doc`.
            * `attention_mask`: [optional] list of attention matrices of length batch_size.
                Each item is a matrix of size `num_sents_in_doc` x `max_i[num tokens in sent i]`
            * `input_lens` [optional]: list of sentence-lengths of length `batch_size`.
                Each item is a tensor of length `num_sentences_in_doc`.


        Returns tuple of (loss, y_preds, y_trues)
         if labels is not None, else
         returns tuple of (None, y_preds, None)
        """
        # batch is list of docs (if only one doc, i.e. not a list, then `vec_or_nones` does the conversion.
        y_pred_lls, y_preds, ys, losses = [], [], [], []
        labels = vec_or_nones(labels, len(input_ids))
        attention_mask = vec_or_nones(attention_mask, len(input_ids))
        input_lens = vec_or_nones(input_lens, len(input_ids))
        add_features = vec_or_nones(add_features, len(input_ids))

        #
        for X, y, a_f, a, s in zip(input_ids, labels, add_features, attention_mask, input_lens):
            if len(X.shape) == 0:
                continue
            loss, y_pred_ll, y = self.predict_one_doc(X, y, a_f, a, s)
            if (not self.is_multitask()) or self.config.do_doc_pred:
                y_pred = y_pred_ll.argmax(dim=1)
            else:
                y_pred = y_pred_ll
            y_pred_lls.append(y_pred_ll)
            y_preds.append(y_pred)
            ys.append(y)
            losses.append(loss)

        if loss is not None:
            loss = torch.sum(loss)
        #
        if (self.config.num_labels_pred_window is None) or (self.config.num_labels_pred_window == 0):
            y_preds = torch.cat(y_preds) # otherwise, `y_preds` is a dict
            ys = torch.cat(ys)

        if not return_lls:
            return loss, y_preds, ys, add_features
        else:
            return loss, y_preds, ys, add_features, y_pred_lls


class Discriminator(LightningMixin, BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)
        #
        self.transformer = SentenceEmbeddingsLayer(*args, **kwargs)
        if self.config.share_label_embeds:
            if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
                self.label_embds = LabelEmbeddings(config=self.config)

        if self.config.separate_heads:
            self.head = nn.ModuleDict()
            self.head['main'] = self.get_head_layer()(*args, **kwargs)
            for i in range(self.config.num_labels_pred_window):
                self.head['backwards %s' % (i + 1)] = self.get_head_layer()(*args, **kwargs)
            for i in range(self.config.num_labels_pred_window):
                self.head['forward %s' % (i + 1)] = self.get_head_layer()(*args, **kwargs)
        else:
            self.head = self.get_head_layer()(*args, **kwargs)

    def get_head_layer(self):
        if self.config.num_contextual_layers == 0:
            return HeadLayerFF if not self.is_multitask() else HeadLayerMultitaskFF
        if self.config.context_layer == 'lstm':
            return HeadLayerLSTM if not self.is_multitask() else HeadLayerMultitaskLSTM
        elif self.config.context_layer == 'gpt2-sentence':
            return HeadLayerTransformer if not self.is_multitask() else HeadLayerMultitaskTransformer

    def get_heads_for_generate(self, labels, label_idx):
        #
        heads = [(label_idx + 1, 'main')]
        n_forward = min(len(labels) - 2 - label_idx, self.config.num_labels_pred_window)
        n_back = min(label_idx + 1, self.config.num_labels_pred_window)
        heads += list(map(lambda x: (label_idx + x + 1, 'forward %s' % x), range(1, n_forward + 1)))
        heads += list(map(lambda x: (label_idx - x + 1, 'backwards %s' % x), range(1, n_back + 1)))
        heads = sorted(heads, key=lambda x: x[0])

        def _get_head_num(head_name):
            m = re.search('\d', head_name)
            if m is not None:
                return int(m[0])
            else:
                return 0

        # weighting
        weighting_vector = []
        for _, head_name in heads:
            position = _get_head_num(head_name)
            if 'forward' in head_name:
                weight = self.config.heads_exp_backoff_right ** position
            elif 'backwards' in head_name:
                weight = self.config.heads_exp_backoff_left ** position
            else:
                weight = 1
            weighting_vector.append(weight)
        weighting_vector = torch.tensor(weighting_vector, device=self.device)
        weighting_vector = weighting_vector / weighting_vector.sum()
        return heads, weighting_vector

    def predict_candidate_batches(self, input_ids, sequence_lens=None, labels=None, label_idx=None):
        """
        For FUDGE: Assume all inputs are tensor matrices of the same size.

        Let this method handle generating the joint probability from all the heads.

        * `input_ids`: num candidates X num words so far
        * `sequence_lens`: lengths of the sequences before
        """
        sent_embs = self.transformer.get_sentence_embedding(
            input_ids=input_ids, attention_mask=None, sequence_lens=sequence_lens, get_last=False
        )
        if isinstance(sent_embs, list):
            tag_probs = []
            if labels is not None and self.config.share_label_embeds:
                label_embs, labels = self.label_embds.get_label_embeddings(labels, 'generate')
            else:
                label_embs = None

            if self.config.separate_heads:
                heads, head_weights = self.get_heads_for_generate(labels, label_idx)

            for s in sent_embs:
                if self.config.separate_heads:
                    preds = []
                    for l_idx, head_key in heads:
                        _, t = self.head[head_key](
                            s, get_last=True, label_embs=label_embs, labels=labels, label_idx=l_idx, get_loss=False,
                            add_features=head_key
                        )
                        preds.append(t)
                    preds = torch.vstack(preds)
                    t = torch.matmul(head_weights, preds)
                else:
                    _, t = self.head(s, get_last=True)
                tag_probs.append(t)
            return torch.vstack(tag_probs)
        else:
            _, tag_probs = self.head(sent_embs, get_last=True)
            return tag_probs

    def pred_from_heads(self, input_ids, sent_embs, labels, label_embs, add_features, label_idx=None, generate=False):
        if add_features is None: # this is hit during generation
            heads, head_weights = self.get_heads_for_generate(labels, label_idx)
            all_tag_preds = []
            all_loss = []
            for _, h in heads:
                loss, tag_preds, _ = self.head[h](
                    sent_embs, labels,
                    label_embs=label_embs,
                    add_features=h,
                    label_idx=label_idx,
                    input_len_eq_one=(input_ids is not None) and (input_ids.shape[1] == 1),
                    generate=generate,
                )
                all_tag_preds.append(tag_preds)
                all_loss.append(loss)
            # return
            all_tag_preds = torch.vstack(all_tag_preds)
            # all_loss = torch.tensor(all_loss, device=self.device)
            return sum(map(mul, head_weights, all_loss)), torch.matmul(head_weights, all_tag_preds), labels

        return self.head[add_features](
            sent_embs, labels,
            label_embs=label_embs,
            add_features=add_features,
            label_idx=label_idx,
            input_len_eq_one=(input_ids is not None) and (input_ids.shape[1] == 1),
            generate=generate
        )

    def _predict_one_doc_one_pass(
            self, input_ids, labels=None, add_features=None,
            attention_mask=None, sequence_lens=None,
            inputs_embeds=None, label_idx=None, generate=False
    ):
        """
        Parameters:
             * `input_ids`: one document tokens (list of sentences. Each sentence is a list of ints.)
             * `labels`: list of y_preds [optional].
             * `attention`: list

        """
        if input_ids is not None and len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        sent_embs = self.transformer.get_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_lens=sequence_lens,
            inputs_embeds=inputs_embeds
        )

        if self.config.share_label_embeds and self.config.use_y:
            if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
                label_embs, labels = self.label_embds.get_label_embeddings(labels, head=add_features)
            else:
                label_embs = None
        else:
            label_embs = None

        if isinstance(sent_embs, list):
            sent_embs = torch.vstack(sent_embs)

        if self.config.separate_heads:
            loss, tag_preds, labels = self.pred_from_heads(
                input_ids=input_ids,
                sent_embs=sent_embs,
                labels=labels,
                label_embs=label_embs,
                add_features=add_features,
                label_idx=label_idx,
                generate=generate
            )

        else:
            output = self.head(
                sent_embs, labels,
                label_embs=label_embs,
                add_features=add_features,
                input_len_eq_one=input_ids.shape[1] == 1 if input_ids is not None else False,
                generate=generate,
                label_idx=label_idx
            )
            if len(output) == 2:
                loss, tag_preds = output
            else:
                loss, tag_preds, labels = output

        return loss, tag_preds, labels



    def predict_one_doc(self, input_ids=None, labels=None, add_features=None,
                        attention_mask=None, sequence_lens=None,
                        inputs_embeds=None, label_idx=None, generate=False
                        ):

        # this part is hit during generation
        if self.config.share_label_embeds and self.config.use_y and (add_features is None):
            if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
                heads, head_weights = self.get_heads_for_generate(labels, label_idx)
                all_tag_preds = []
                all_loss = []
                for h_idx, h in heads:
                    loss, tag_preds, _ = self._predict_one_doc_one_pass(
                        input_ids, labels, h, attention_mask,
                        sequence_lens, inputs_embeds, label_idx,
                        generate=generate
                    )
                    all_tag_preds.append(tag_preds) # todo: make this a weighted prediction sum for the generator
                    all_loss.append(loss) # todo: make this a weighted loss for the generator

                all_tag_preds = torch.vstack(all_tag_preds)
                return sum(map(mul, head_weights, all_loss)), torch.matmul(head_weights, all_tag_preds), labels

        else:
            return self._predict_one_doc_one_pass(
                input_ids,
                labels,
                add_features,
                attention_mask,
                sequence_lens,
                inputs_embeds,
                label_idx,
                generate=generate
            )

