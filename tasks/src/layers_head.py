import torch
from torch import nn as nn

from tasks.src.layers_attention import DocEmbeddingForDocLabelClass
from tasks.src.layers_classification import MultiClassMixin, MultiTaskMultiClassMixin, BinaryMixin
from tasks.src.layers_contextualizing import FFContextMixin, BiLSTMContextMixin, TransformerContextMixin
from tasks.src.layers_embeddings import EmbeddingHandlerMixin
from tasks.src.layers_label_embedding import LabelEmbeddings
from tasks.src.utils_general import get_config


class HeadBase(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)

    def forward(self, sent_embs, labels=None, *args, **kwargs):
        """
        Parameters:
            * `sent_embs`: list of sentence embeddings.
            * `labels`: list of labels.

        If labels provided, returns (loss, prediction). Else, returns (None, prediction).
        """
        sent_embs = self.get_contextualized_embeddings(sent_embs)
        augmented_embs = self.transform_sentence_embeddings(sent_embs)
        return self.classification(augmented_embs, labels)


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