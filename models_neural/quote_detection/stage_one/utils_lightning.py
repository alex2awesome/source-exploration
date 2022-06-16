import pytorch_lightning as pl

from discriminator.utils_metrics import (
    get_classification_report_metric,
    format_classification_report,
    Entropy,
    MaxCount
)
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from discriminator.utils_general import get_config

adam_beta1, adam_beta2, adam_epsilon = .9, .999, 1e-08

class LightningBase(pl.LightningModule):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.config = get_config(config=config, kwargs=kwargs)

        self.save_hyperparameters(self.config.to_dict(), ignore=[
            'train_data_file_s3', 'log_all_metrics', 'save_model', 'do_train', 'do_eval', 'notes', 'local', 'use_cpu', 'transformer_model_name',
            'model_name', 'pretrained_cache_dir', 'main_data_file', 'pad_id', 'num_steps_per_epoch',
            'total_steps'
        ])

        #####
        # metrics
        dist_sync_on_step = kwargs.get('accelerator') == 'dp'
        self.log_all_metrics = self.config.log_all_metrics
        self.training_report = get_classification_report_metric(self.config, dist_sync_on_step)
        self.validation_report = get_classification_report_metric(self.config, dist_sync_on_step)
        if not self.is_multitask():
            self.entropy = Entropy(num_classes=self.config.num_output_tags)
            self.max_count = MaxCount(num_classes=self.config.num_output_tags)
        self.hp_metric_list = []  # to store f1-scores and then take the max

    def is_multitask(self):
        if self.config.separate_heads:
            return False

        return self.config.do_multitask or \
               (self.config.num_labels_pred_window is not None and
               self.config.num_labels_pred_window != 0)

    def _format_step_output(self, loss, y_pred, y_true, add_features, batch):
        output = {'loss': loss, 'y_pred': y_pred}
        y_true = y_true if y_true is not None else batch['labels']
        if isinstance(y_true, list):
            y_true = torch.cat(y_true)
        else:
            if len(y_true.shape) == 0:
                y_true = y_true.unsqueeze(0)
        output['y_true'] = y_true
        if self.config.separate_heads:
            output['head'] = add_features
        return output

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true, add_features = self.forward(**batch)
        self.log('Training Loss', loss)
        output = self._format_step_output(loss, y_pred, y_true, add_features, batch)
        return output

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true, add_features = self.forward(**batch)
        self.log('Validation loss', loss)
        output = self._format_step_output(loss, y_pred, y_true, add_features, batch)
        return output

    def training_step_end(self, batch_parts):
        # if run on multi-GPUs, this is a list(?)
        if isinstance(batch_parts, list):
            for batch in batch_parts:
                self.training_report(batch['y_pred'], batch['y_true'])
                if not self.is_multitask() and (not self.config.separate_heads):
                    self.entropy(batch['y_pred'])
                    self.max_count(batch['y_pred'])
            return sum(map(lambda x: x['loss'], batch_parts))
        # otherwise, it's a float(?)
        else:
            if 'head' in batch_parts:
                self.training_report(batch_parts['y_pred'], batch_parts['y_true'], head=batch_parts.get('head'))
            else:
                self.training_report(batch_parts['y_pred'], batch_parts['y_true'])
            if not self.is_multitask() and (not self.config.separate_heads):
                self.max_count(batch_parts['y_pred'])
                self.entropy(batch_parts['y_pred'])
            return batch_parts['loss']

    def validation_step_end(self, batch_parts):
        if isinstance(batch_parts, list):
            for batch in batch_parts:
                self.validation_report(batch['y_pred'], batch['y_true'])
                if not self.is_multitask():
                    self.entropy(batch['y_pred'])
                    self.max_count(batch['y_pred'])
        else:
            # print('DEVICES:')
            # print('Y_PRED: %s' % str(batch_parts['y_pred'].device))
            # print('Y_TRUE: %s' % str(batch_parts['y_true'].device))
            if 'head' in batch_parts:
                self.validation_report(batch_parts['y_pred'], batch_parts['y_true'], head=batch_parts.get('head'))
            else:
                self.validation_report(batch_parts['y_pred'], batch_parts['y_true'])
            if not self.is_multitask() and (not self.config.separate_heads):
                self.entropy(batch_parts['y_pred'])
                self.max_count(batch_parts['y_pred'])

    def training_epoch_end(self, outputs):
        report = self.training_report.compute()
        if self.log_all_metrics:
            self.log_dict(format_classification_report(report, 'Training', config=self.config))
        self.training_report.reset()

    def validation_epoch_end(self, outputs):
        report = self.validation_report.compute()
        if self.log_all_metrics:
            self.log_dict(format_classification_report(report, 'Validation', config=self.config))
        self.validation_report.reset()
        if not self.is_multitask() and (not self.config.separate_heads):
            self.log('f1 macro', report['Macro F1'])
            self.log('f1 weighted', report['Weighted F1'])
            self.hp_metric_list.append(report['Macro F1'])
            self.log('hp_metric', max(self.hp_metric_list))
            self.log('entropy', self.entropy.compute())
            self.log('max count', self.max_count.compute())
            self.entropy.reset()
            self.max_count.reset()

    def on_validation_end(self):
        if not self.is_multitask() and not self.config.separate_heads:
            try:
                self.log('hp_metric', max(self.hp_metric_list))
            except MisconfigurationException:
                pass


class LightningMixin(LightningBase):
    """
    Contains logic for optimization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)

        #
        self.lr = self.config.learning_rate
        self.num_warmup_steps = self.config.num_warmup_steps
        self.dataset_size = self.config.num_steps_per_epoch

    # optimization
    def _lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        return 1.0

    def _lr_lambda_linear(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        num = self.num_training_steps - current_step
        denom = self.num_training_steps - self.num_warmup_steps
        num = float(max(0, num))
        denom = float(max(1, denom))
        return num / denom

    def configure_optimizers(self):
        self.num_training_steps = self.dataset_size * self.trainer.max_epochs
        optimizer_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        }
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **optimizer_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda_linear),
                'interval': 'step',
            }
        }


