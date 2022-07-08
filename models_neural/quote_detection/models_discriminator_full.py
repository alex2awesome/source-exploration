import torch.nn as nn
import transformers
from packaging import version

from models_neural.src.layers_head import HeadLayerFF, HeadLayerLSTM, HeadLayerTransformer, \
    HeadLayerMultitaskFF, HeadLayerMultitaskLSTM, HeadLayerMultitaskTransformer
from models_neural.src.layers_label_embedding import LabelEmbeddings
from models_neural.src.layers_sentence_embedding import SentenceEmbeddingsLayer

if version.parse(transformers.__version__) == version.parse('3.0.2'):
    pass
else: # transformers: version 4.0
    pass
import torch
from operator import mul

from models_neural.src.utils_lightning import LightningMixin
from models_neural.src.utils_general import (
    get_config, vec_or_nones, _get_head_num
)


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
        #
        if self.config.separate_heads:
            self.head = nn.ModuleDict()
            self.head['main'] = self.get_head_layer()(*args, **kwargs)
            for i in range(self.config.num_labels_pred_window):
                self.head['backwards %s' % (i + 1)] = self.get_head_layer()(*args, **kwargs)
            for i in range(self.config.num_labels_pred_window):
                self.head['forward %s' % (i + 1)] = self.get_head_layer()(*args, **kwargs)
        #
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

