import torch
import math
from transformers import BertModel, GPT2LMHeadModel, RobertaModel
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from discriminator.utils_metrics import (
    get_classification_report_metric,
    format_classification_report,
    Entropy,
    MaxCount
)
from discriminator.utils_general import format_layer_freezes
from util.utils_general import reformat_model_path, format_local_vars
from discriminator.layers_attention import DocLevelSelfAttention
from torch import nn

torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."


class ClassificationHead(nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, hidden_dim, dropout):
        super().__init__()
        self.class_size = class_size
        self.dropout = nn.Dropout(dropout)
        self.mlp = torch.nn.Linear(hidden_dim, class_size)

    def forward(self, hidden_state):
        hidden_state = self.dropout(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class BaseDiscriminator(pl.LightningModule):
    """Transformer encoder followed by a Classification Head"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = kwargs.get('lr')
        self.num_warmup_steps = kwargs.get('num_warmup_steps', 0)
        self.batch_size = kwargs.get('batch_size')
        self.dataset_size = kwargs.get('dataset_size')

        if not kwargs.get('class_size'):
            raise ValueError("must specify class_size")

        # load pretrained model
        self.get_pretrained_model(
            model_type=kwargs.get('model_type'),
            model_path_on_disk=reformat_model_path(kwargs.get('pretrained_model_path'))
        )

        # sentence embedding method
        self.sentence_embedding_method = kwargs.get('sentence_embedding_method')

        # classifier head type
        self.get_classification_head(
            hidden_dim=self.get_final_layer_size(),
            classifier_head_type=kwargs.get('classifier_head_type'),
            class_size=kwargs.get('class_size'),
            dropout=kwargs.get('dropout'),
        )

        #########
        # freeze layers
        self.model_type = kwargs.get('model_type')
        self.freeze_transformer = kwargs.get('freeze_transformer')
        self.freeze_embedding_layer = kwargs.get('freeze_embedding_layer')
        self.freeze_encoder_layers = kwargs.get('freeze_encoder_layers')
        self._freeze_encoder_layers()

        ##############
        # additional embeddings
        # headline
        self.use_headline_emb = kwargs.get('use_headline_emb')
        # doc embeddings
        self.use_doc_emb = kwargs.get('use_doc_emb')
        self.doc_attention = DocLevelSelfAttention(
            hidden_dim=self.get_final_layer_size(),
            dropout=kwargs.get('dropout'),
        )
        # positional embeddings
        self.use_positional_emb = kwargs.get('use_positional_emb')
        self.use_sinusoidal_emb = kwargs.get('use_sinusoidal_emb')
        self.max_positional_embeddings = kwargs.get('max_positional_embeddings', 40)
        self.pad_token_id = kwargs.get('pad_token_id')
        if not self.use_sinusoidal_emb:
            self.param_max_positional = nn.Parameter(torch.tensor(self.max_positional_embeddings), requires_grad=False)
            self.param_default_max_positional = nn.Parameter(torch.tensor(self.max_positional_embeddings - 1), requires_grad=False)
            self.position_embeddings = nn.Embedding(self.max_positional_embeddings, self.get_final_layer_size())
        else:
            from fairseq.modules import SinusoidalPositionalEmbedding
            self.position_embeddings = SinusoidalPositionalEmbedding(
                self.get_final_layer_size(),
                self.pad_token_id,
                self.max_position_embeddings
            )

        #####
        # metrics
        self.log_all_metrics = kwargs.get('log_all_metrics', False)
        self.training_report = get_classification_report_metric(
            num_classes=kwargs.get('class_size'), dist_sync_on_step=kwargs.get('accelerator') == 'dp'
        )
        self.validation_report = get_classification_report_metric(
            num_classes=kwargs.get('class_size'), dist_sync_on_step=kwargs.get('accelerator') == 'dp'
        )
        self.entropy = Entropy(num_classes=kwargs.get('class_size'))
        self.max_count = MaxCount(num_classes=kwargs.get('class_size'))
        self.hp_metric_list = [] # to store f1-scores and then take the max

    def get_pretrained_model(self, model_type, model_path_on_disk):
        # get pretrained model
        if model_type == "gpt2":
            self.encoder = GPT2LMHeadModel.from_pretrained(model_path_on_disk)
            self.embed_size = self.encoder.transformer.config.hidden_size
        elif model_type == "bert":
            self.encoder = BertModel.from_pretrained(model_path_on_disk)
            self.embed_size = self.encoder.config.hidden_size
        elif model_type == 'roberta':
            self.encoder = RobertaModel.from_pretrained(model_path_on_disk)
            self.embed_size = self.encoder.config.hidden_size
        else:
            raise ValueError(
                "{} model not yet supported".format(model_type)
            )

    def get_sentence_embedding(self, input_ids, attention_mask):
        hidden, mask = self._get_word_embeddings(input_ids, attention_mask)
        if self.sentence_embedding_method == 'average':
            return self._avg_representation(hidden, mask)
        if self.sentence_embedding_method == 'cls':
            return self._cls_token(hidden)
        else:
            print('SENTENCE EMBEDDING METHOD %s not in {average, cls}')

    def _init_classification_head_weights(self):
        nn.init.xavier_uniform_(self.classifier_head.state_dict()['mlp.weight'])
        self.classifier_head.mlp.bias.data.fill_(0)

    def get_classification_head(self, hidden_dim, classifier_head_type, class_size, dropout):
        if classifier_head_type == 'feed-forward':
            self.classifier_head = ClassificationHead(
                class_size=class_size,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        self._init_classification_head_weights()

    def _get_word_embeddings(self, input_ids, attention_mask):
        additive_mask = (
            attention_mask
              .unsqueeze(2)
              .repeat(1, 1, self.embed_size)
              .float()
              .detach()
        )
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            hidden, _ = self.encoder.transformer(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # for bert
            hidden, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return hidden, additive_mask

    def _avg_representation(self, hidden, mask):
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def _cls_token(self, hidden):
        cls_embeddings = hidden[:, 0, :]
        return cls_embeddings

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self.shared_step(batch)
        self.log('Training Loss', loss)
        return {'loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self.shared_step(batch)
        self.log('Validation loss', loss)
        return {'loss': loss, 'y_pred': y_pred, "y_true": y_true}

    def training_step_end(self, batch_parts):
        # if run on multi-GPUs, this is a list(?)
        if isinstance(batch_parts, list):
            for batch in batch_parts:
                self.training_report(batch['y_pred'], batch['y_true'])
                self.entropy(batch['y_pred'])
                self.max_count(batch['y_pred'])
            return sum(map(lambda x: x['loss'], batch_parts))
        # otherwise, it's a float(?)
        else:
            self.training_report(batch_parts['y_pred'], batch_parts['y_true'])
            self.max_count(batch_parts['y_pred'])
            self.max_count(batch_parts['y_pred'])
            return batch_parts['loss']

    def validation_step_end(self, batch_parts):
        if isinstance(batch_parts, list):
            for batch in batch_parts:
                self.validation_report(batch['y_pred'], batch['y_true'])
                self.entropy(batch['y_pred'])
                self.max_count(batch['y_pred'])
        else:
            self.validation_report(batch_parts['y_pred'], batch_parts['y_true'])
            self.entropy(batch_parts['y_pred'])
            self.max_count(batch_parts['y_pred'])

    def training_epoch_end(self, outputs):
        report = self.training_report.compute()
        if self.log_all_metrics:
            self.log('Training Report', format_classification_report(report))
        self.training_report.reset()

    def validation_epoch_end(self, outputs):
        report = self.validation_report.compute()
        if self.log_all_metrics:
            self.log('Validation Report', format_classification_report(report))
        self.log('f1 macro', report['Macro F1'])
        self.log('entropy', self.entropy.compute())
        self.log('max count', self.max_count.compute())
        self.hp_metric_list.append(report['Macro F1'])
        self.validation_report.reset()
        self.entropy.reset()
        self.max_count.reset()

    def on_validation_end(self):
        self.log('hp_metric', max(self.hp_metric_list))
    ##
    # embedding augmentation methods
    #
    def _get_doc_embedding(self, cls):
        return self.doc_attention(cls)

    def _get_headline_embedding(self, cls):
        '''Given a sequence of [CLS] tokens, separate a headline embedding matrix and the rest of sentences.'''
        headline_embedding = cls[0]
        cls = cls[1:]
        headline_embedding = headline_embedding.unsqueeze(0).expand(cls.size())
        return headline_embedding, cls

    def _get_position_embeddings(self, hidden_embs):
        # get position embeddings
        if not self.use_sinusoidal_emb:
            position_ids = torch.arange(len(hidden_embs), dtype=torch.long, device=hidden_embs.device)
            # assign long sequences the same embedding
            position_ids = position_ids.where(position_ids < self.param_max_positional, self.param_default_max_positional)
            position_embeddings = self.position_embeddings(position_ids)
        else:
            position_embeddings = self.position_embeddings(hidden_embs[:, [0]]).squeeze()
        return position_embeddings

    def get_all_embeddings(self, hidden_embs):
        # headline embeddings
        if self.use_headline_emb:
            headline_embedding, hidden_embs = self._get_headline_embedding(hidden_embs)
        else:
            headline_embedding, hidden_embs = None, hidden_embs
        # positional embeddings
        position_embeddings = self._get_position_embeddings(hidden_embs) if self.use_positional_emb else None
        # document embeddings
        doc_embedding = self._get_doc_embedding(hidden_embs) if self.use_doc_emb else None
        return hidden_embs, headline_embedding, position_embeddings, doc_embedding

    # freeze pretrained model layers
    def _freeze_encoder_layers(self):
        # freeze whole transformer
        if self.freeze_transformer:
            for p in self.encoder.parameters():
                p.requires_grad = False
            return

        # freeze embedding layer
        if self.freeze_embedding_layer:
            if self.model_type == 'gpt2':
                for p in self.encoder.wte.parameters():
                    p.requires_grad = False
            else:
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = False

        # freeze encoding layers
        if self.freeze_encoder_layers:
            layers_to_freeze = format_layer_freezes(self.freeze_encoder_layers)
            for layer in layers_to_freeze:
                if self.model_type == 'gpt2':
                    for p in self.encoder.transformer.h[layer].parameters():
                        p.requires_grad = False
                else:
                    for p in self.encoder.encoder.layer[layer].parameters():
                        p.requires_grad = False


########
# Discriminators
class BaselineDiscriminator(BaseDiscriminator):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            model_notes="Baseline Non-Sequential Discriminator",
            accelerator=None,
            # intermediate layer types
            classifier_head_type='feed-forward',
            sentence_embedding_method="average",
            pretrained_model_path="gpt2-medium",
            model_type='gpt2',
            # params
            class_size=None,
            dataset_size=None,
            lr=0.0001,
            batch_size=1,
            dropout=.1,
            # freeze layers
            freeze_transformer=False,
            freeze_embedding_layer=False,
            freeze_encoder_layers=False,
            # use embedding enhancements
            use_headline_emb=False,
            use_doc_emb=False,
            use_positional_emb=False,
            use_sinusoidal_emb=False,
            **kwargs
    ):
        super().__init__(**format_local_vars(locals()))

    def get_final_layer_size(self):
        return self.embed_size

    def shared_step(self, batch):
        """
        Step that's shared between training loop and validation loop. Contains nonsequence-specific processing,
        so we're keeping it in the child class.

        Returns tuple of (loss, y_preds, y_trues)
        """
        x, y, attention_mask = batch
        y_pred_ll = self.forward(x)
        y_pred = y_pred_ll.argmax(dim=1, keepdim=True).flatten()
        return F.nll_loss(y_pred_ll, y), y_pred, y

    def forward(self, x, attention_mask):
        hidden = self.get_sentence_embedding(x, attention_mask)
        logits = self.classifier_head(hidden)
        probs = F.log_softmax(logits, dim=-1)
        return probs

    ## optimization
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=math.ceil(self.dataset_size / self.batch_size)
                ),
                'interval': 'step',
            }
        }