import torch
from torch import nn as nn

from models_neural.src.layers_attention import DocEmbeddingForDocLabelClass
from models_neural.src.layers_classification import MultiClassMixin, MultiTaskMultiClassMixin, BinaryMixin
from models_neural.src.layers_contextualizing import FFContextMixin, BiLSTMContextMixin, TransformerContextMixin
from models_neural.src.layers_embeddings import EmbeddingHandlerMixin
from models_neural.src.layers_label_embedding import LabelEmbeddings
from models_neural.src.utils_general import get_config


class HeadBase(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)
        if getattr(self.config, 'do_doc_pred', False):
            self.doc_embeddings = DocEmbeddingForDocLabelClass(*args, **kwargs)
        if getattr(self.config, 'use_y', False) and not getattr(self.config, 'share_label_embeds', False):
            if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
                self.label_embedding_layer = LabelEmbeddings(config=self.config)

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
                label_embs, labels = self.label_embedding_layer(labels, head=add_features)
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
        if getattr(self.config, 'use_y', False):
            label_embs, labels = self.handle_y_embeddings(labels, label_embs, label_idx, add_features)
        augmented_embs = self.handle_x_embeddings(sent_embs, add_features, get_last, generate, *args, **kwargs)
        if get_loss:
            return self.classification(augmented_embs, labels, label_embs)
        else:
            return self.classification(augmented_embs, None, label_embs)


class HeadLayerBinaryFF(BinaryMixin, FFContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HeadLayerBinaryLSTM(BinaryMixin, BiLSTMContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HeadLayerBinaryTransformer(BinaryMixin, TransformerContextMixin, HeadBase, EmbeddingHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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