from transformers import PretrainedConfig, TrainingArguments, AutoConfig
from dataclasses import dataclass
import os
import json

@dataclass(init=False)
class TransformersConfig(PretrainedConfig):
    ### this is only for record keeping
    ## dropout
    dropout: float
    transformer_hidden_dropout_prob: float ## default = .1
    transformer_attention_probs_dropout_prob: float ## default = .1
    ## general model params
    freeze_transformer: bool
    freeze_embedding_layer: bool
    freeze_encoder_layers: list
    freeze_pooling_layer: bool
    embedding_dim: int ## default set by TransformerConfig in the code
    hidden_dim: int
    num_output_tags: int
    # bi-lstm
    bidirectional: bool
    num_lstm_layers: int
    # training parameters
    random_split: bool  ## whether do to a random split or use the dataset's natural split
    train_perc: float  ## how much of the training set to use
    ## embedding augmentations
    use_positional: bool
    max_position_embeddings: int ## for positional embeddings, specifies how far out we go.
    use_doc_emb: bool
    doc_embed_arithmetic: bool
    concat_headline: bool ## actually use the headline embedding

    # params with defaults
    # other
    use_cpu: bool = False

    def __init__(self, cmd_args=None, *args, **kwargs):
        if cmd_args is not None:
            # pass in all args
            for k in cmd_args.__dict__:
                self.__dict__[k] = cmd_args.__dict__[k]

        # make sure default annotations are also in the state dict (for `.to_json_string()`)
        for k in self.__annotations__:
            if k not in self.__dict__:
                try:
                    self.__dict__[k] = self.__getattribute__(k)
                except AttributeError:
                    pass

        if cmd_args is not None:
            # custom processing of input
            self.notes = cmd_args.notes.replace('_', ' ')
            self.freeze_encoder_layers = list(map(int, cmd_args.freeze_encoder_layers))
            self.transformer_hidden_dropout_prob = cmd_args.dropout
            self.transformer_attention_probs_dropout_prob = cmd_args.dropout
            self.embedding_dim = 0 ## default set by transformer in the code

        # handle kwargs for from_dict reconstruction
        for k, v in kwargs.items():
            self.__dict__[k] = v

        if hasattr(self, 'pretrained_cache_dir'):
            if os.environ.get('env') != 'bb':
                if self.pretrained_cache_dir is not None and self.pretrained_cache_dir.startswith('./'):
                    self.pretrained_cache_dir = self.pretrained_cache_dir[2:]

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        config_dict.pop('args', '')
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __repr__(self):
        return str(self.to_dict())

    def __getattr__(self, item):
        """Only called if the attribute doesn't actually exist."""
        if item in ('pruned_heads'):
            raise AttributeError()

        return None

    def __setattr__(self, name, value):
        self.__dict__[name] = value


###
### configs
###
training_args = TrainingArguments(
    output_dir='tmp',
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    dataloader_drop_last=False,
    num_train_epochs=15,
    ###
    save_steps=0, # don't save model. remove this for a real run.
    logging_steps=400,
    eval_steps=3000, ## set this so that you evaluate at the end of every epoch
    # evaluate_during_training=True,
)


def get_transformer_config(pretrained_cache_dir):
    transformer_config = AutoConfig.from_pretrained(pretrained_cache_dir)
    # transformer_config.hidden_dropout_prob = config.transformer_hidden_dropout_prob
    # transformer_config.attention_probs_dropout_prob = config.transformer_attention_probs_dropout_prob
    return transformer_config