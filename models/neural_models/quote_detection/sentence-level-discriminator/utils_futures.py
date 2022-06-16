from torch import nn
import transformers
if transformers.__version__ == '3.0.2':
    from transformers.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model
    from torch.nn import CrossEntropyLoss, MSELoss
    import sys, os
    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    from utils_general import freeze_all_params
    import torch
    try:
        from transformers.utils import logging
        logger = logging.get_logger(__name__)
    except:
        import logging
        logger = logging.getLogger(__name__)


    class GPT2ForSequenceClassification(GPT2PreTrainedModel):
        _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.transformer = GPT2Model(config)
            freeze_all_params(self.transformer)
            self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

            self.init_weights()

            # Model parallel
            self.model_parallel = False
            self.device_map = None

        def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
        ):
            r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            try: # version 4.8.0
                transformer_outputs = self.transformer(
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            except: # version 3.0.2
                transformer_outputs = self.transformer(
                    input_ids,
                    past=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            hidden_states = transformer_outputs[0]
            logits = self.score(hidden_states)

            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape[:2]
            else:
                batch_size, sequence_length = inputs_embeds.shape[:2]

            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                else:
                    sequence_lengths = -1
                    logger.warning(
                        f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                        f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                    )

            pooled_logits = logits[range(batch_size), sequence_lengths]

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(pooled_logits.view(-1), labels.to(self.dtype).view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

else:
    from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification