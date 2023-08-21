import torch.nn as nn
from typing import Optional, List, Dict, Union, Any

from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, RobertaConfig, RobertaModel, PreTrainedModel, BertPreTrainedModel

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.special import expit

###############################
# model components
def freeze_hf_model(model, freeze_layers, model_type):
    def freeze_all_params(subgraph):
        for p in subgraph.parameters():
            p.requires_grad = False

    if model_type == 'bert':
        layers = model.encoder.layer
    else:
        layers = model.transformer.h

    if freeze_layers is not None:
        for layer in freeze_layers:
            freeze_all_params(layers[layer])


class QAModel(BertPreTrainedModel):
    def __init__(self, config, hf_model=None):
        super().__init__(config)
        self.config = config

        base_model = AutoModel.from_config(config) if hf_model is None else hf_model
        QAModel.base_model_prefix = base_model.base_model_prefix
        QAModel.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)  # setattr(x, 'y', v) is equivalent to ``x.y = v''

        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def position_decay(self, start_positions, end_positions, global_step, max_steps):
        if getattr(self.config, 'loss_window', None) is not None:
            if max_steps > 0 and len(start_positions.shape) > 1:
                exp_decay = global_step / max_steps * 10
                start_positions = start_positions ** exp_decay
                end_positions = end_positions ** exp_decay
        return start_positions, end_positions

    def forward(
            self,
            input_ids,
            token_type_ids,
            start_positions=None,
            end_positions=None,
            attention_mask=None,
            *args,
            **kwargs
    ):

        outputs = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        word_embs = outputs[0]

        logits = self.qa_outputs(word_embs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_positions, end_positions = self.position_decay(
                start_positions, end_positions,
                global_step=kwargs.get('global_step', 0), max_steps=kwargs.get('max_steps', 0)
            )
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

    def post_init(self):
        # during prediction, we don't have to pass this in
        if hasattr(self.config, 'freeze_layers'):
            base_model = getattr(self, self.base_model_prefix)
            freeze_hf_model(base_model, freeze_layers=self.config.freeze_layers,
                            model_type=base_model.base_model_prefix)


class QAModelWithSalience(BertPreTrainedModel):
    def __init__(self, config, hf_model=None):
        super().__init__(config)

        base_model = AutoModel.from_config(config) if hf_model is None else hf_model
        QAModelWithSalience.base_model_prefix = base_model.base_model_prefix
        QAModelWithSalience.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)

        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.salience_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def get_forward_logits(self, input_ids, attention_mask, cls_head, token_type_ids=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        word_embs = outputs[0]

        logits = cls_head(word_embs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            start_positions=None,
            end_positions=None,
            attention_mask=None,
            *args,
            **kwargs
    ):
        if token_type_ids is not None:
            ref_start_logits, ref_end_logits = self.get_forward_logits(
                input_ids, attention_mask, self.qa_outputs, token_type_ids
            )
            if len(input_ids) == 1:
                input_ids = input_ids[token_type_ids == 0].unsqueeze(0)
            else:
                raise ValueError('Need to be able to handle batches > 1.')
            ref_start_logits = ref_start_logits[:, : input_ids.shape[1]]
            ref_end_logits = ref_end_logits[:, : input_ids.shape[1]]

        start_logits, end_logits = self.get_forward_logits(input_ids, attention_mask, self.salience_outputs)
        if token_type_ids is not None:
            start_logits = start_logits + ref_start_logits
            end_logits = end_logits + ref_end_logits

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output

    def post_init(self):
        # during prediction, we don't have to pass this in
        if hasattr(self.config, 'freeze_layers'):
            base_model = getattr(self, self.base_model_prefix)
            freeze_hf_model(base_model, freeze_layers=self.config.freeze_layers,
                            model_type=base_model.base_model_prefix)
