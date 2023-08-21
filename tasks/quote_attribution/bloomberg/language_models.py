try: # transformers: version 3.0.2
    from transformers.modeling_gpt2 import Block as GPT2LMHeadModel
except: # transformers: version 4.0
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    from transformers.models.roberta.modeling_roberta import RobertaForCausalLM

from typing import Tuple, Any, List, Dict
import torch
from torch import nn
from models_neural.src.utils_lightning import LightningOptimizer, LightningLMSteps
from models_neural.src.utils_general import format_layer_freezes, freeze_all_params, get_config
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer


class BaseLMModel(LightningOptimizer, LightningLMSteps):
    def __init__(self, model=None, model_type='gpt2', *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        self.model_type = model_type
        super().__init__(*args, **kwargs)
        self.hf_model = model
        if kwargs.get('loading_from_checkpoint'):
            self.hf_model = self.get_base_hf_model()
        if self.hf_model is not None:
            self.freeze_encoder_layers()

    def get_base_hf_model(self):
        """Only used if we're loading from a checkpoint..."""
        if self.model_type == 'gpt2':
            return GPT2LMHeadModel(config=self.config)
        else:
            return RobertaForCausalLM(config=self.config)

    def freeze_encoder_layers(self):
        # freeze whole transformer
        if self.config.freeze_transformer:
            freeze_all_params(self.hf_model)

        # freeze embedding layer
        if self.config.freeze_embedding_layer:
            if self.config.model_type == 'gpt2':
                freeze_all_params(self.hf_model.transformer.wte)
            else:
                freeze_all_params(self.hf_model.embeddings)

        # freeze encoding layers
        if self.config.freeze_encoder_layers:
            layers_to_freeze = format_layer_freezes(self.config.freeze_encoder_layers)
            for layer in layers_to_freeze:
                if self.config.model_type == 'gpt2':
                    freeze_all_params(self.hf_model.transformer.h[layer])
                else:
                    freeze_all_params(self.hf_model.encoder.layer[layer])


class LMModel(BaseLMModel):
    def __init__(self, model=None, model_type='gpt2', *args, **kwargs):
        super().__init__(model=model, model_type=model_type, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.hf_model.forward(*args, **kwargs)

    def get_lmhead_logits_and_past_and_hidden(
            self, input_ids=None, attention_mask=None, past_key_values=None, input_embeds=None
    ):
        """Pass-through method, here for convenience (Used in the generator_pplm.)"""
        assert (input_ids is not None) or (input_embeds is not None)
        # fix input
        if past_key_values is not None:
            if not isinstance(past_key_values[0], tuple):
                past_key_values = tuple(list(map(lambda l: (l[[0]], l[[1]]), past_key_values)))

        logits, past, all_hidden = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            use_cache=True,
            output_hidden_states=True,
            return_dict=False
        )

        # versioning shift
        past = tuple(list(map(lambda l: torch.cat(l), past)))
        return logits, past, all_hidden


class SourceModel(BaseLMModel):
    """
    Similar to the `LMModel` except it does language generation using a sentence-level embedding
    that also has positional information on top of it.
    """
    def __init__(self, model=None, model_type='gpt2', *args, **kwargs):
        super().__init__(model=model, model_type=model_type, *args, **kwargs)
        pass

class GPT2Wrapper(LightningModule):
    """A thin wrapper around HuggingFace GPT2 implementation to adapt the forward and loss functions to LIT's trainer.
    Args:
        hf_model: the HuggingFace GPT2 model to wrap.
        pad_index: the token index used for padding the input.
    """

    def __init__(self, hf_model: nn.Module, pad_index: int):
        """Initialize the wrapper."""
        super().__init__()
        self.hf_model = hf_model
        self.pad_index = pad_index
        # self.perplexity = Perplexity(device=self.device)

    def forward(self, input_ids: torch.Tensor):
        """Run a forward pass on the input.
        Args:
            input_ids: Tensor of input token indexes.
        """
        return self.hf_model(
            input_ids=input_ids,
            attention_mask=torch.eq(input_ids, self.pad_index).to(torch.long),
        ).logits

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor, return_outputs: bool = False):
        """Run a forward pass on the input.
        Args:
            input_ids: Tensor of input token indexes.
            labels: Tensor of expected token indexes, with non-target tokens masked with value -100.
            return_outputs: boolean indicating whether to return in model logits. If False, the second element in the output tuple will be None.
        """
        results = self.hf_model(
            input_ids=input_ids,
            attention_mask=torch.eq(input_ids, self.pad_index).to(dtype=torch.long, device=input_ids.device),
            labels=labels,
        )
        return (results.loss, results.logits if return_outputs else None)

    def _take_step(self, batch: Tuple[Any, torch.Tensor], mode: str) -> Dict[str, Any]:
        input, labels = batch['input_ids'], batch['labels']
        step_output = {}
        output = None
        loss, _ = self.loss(labels=labels, return_outputs=False, input_ids=input)
        self.log(
            "{}_loss".format(mode), loss, on_step=False, on_epoch=True, logger=True
        )
        step_output["loss"] = loss
        step_output['input_ids'] = input

        return step_output

    # The <mode>_step and <mode>_epoch_end methods are called automatically by
    # the Lightning Trainer at the corresponding point during training or test.
    def training_step(self, batch, batch_idx):
        """Takes a training step.
        ***Called automatically by trainer, should not be called by user.***
        """
        return self._take_step(batch, "training")

    def validation_step(self, batch, batch_idx):
        """Takes a validation step.
        ***Called automatically by trainer, should not be called by user.***
        """
        return self._take_step(batch, "validation")

    def configure_optimizers(self) -> optim.Optimizer:
        """Creates optimizers for the model.
        ***Called automatically by trainer, should not be called by user.***
        """
        return optim.AdamW(
            self.parameters(),
            lr=5e-5,
            weight_decay=0.0
        )

