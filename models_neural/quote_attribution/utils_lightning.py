import pytorch_lightning as pl
from torchmetrics import F1

from models_neural.quote_attribution.utils_metrics import BasicAccuracy, SequenceF1


class LightningClassificationSteps(pl.LightningModule):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        dist_sync_on_step = kwargs.get('accelerator') == 'dp'
        # assume this is just for the binary classification task.
        self.training_f1 = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
        self.validation_f1 = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)

    def training_step(self, batch, batch_idx):
        loss, pred, label = self.forward(**batch)
        self.training_f1(pred, label.to(int))
        self.log('Training Loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, pred, label = self.forward(**batch)
        self.validation_f1(pred, label.to(int))
        self.log('Validation loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        f1_score = self.training_f1.compute()
        self.log('Training f1 score', f1_score)
        self.training_f1.reset()

    def validation_epoch_end(self, outputs):
        f1_score = self.validation_f1.compute()
        self.log('Validation f1 score', f1_score)
        self.validation_f1.reset()


class LightningQASteps(pl.LightningModule):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.training_accuracy = BasicAccuracy()
        self.validation_accuracy = BasicAccuracy()
        self.training_f1 = SequenceF1()
        self.validation_f1 = SequenceF1()

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss, start_logits, end_logits = output[0], output[1], output[2]
        start_label, end_label = batch['start_positions'], batch['end_positions']

        self.log('Training Loss', loss)
        self.training_accuracy((start_logits, end_logits), (start_label, end_label))
        self.training_f1((start_logits, end_logits), (start_label, end_label))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss, start_logits, end_logits = output[0], output[1], output[2]
        start_label, end_label = batch['start_positions'], batch['end_positions']

        self.log('Validation loss', loss)
        self.validation_accuracy((start_logits, end_logits), (start_label, end_label))
        self.validation_f1((start_logits, end_logits), (start_label, end_label))
        return {'loss': loss}

    def training_step_end(self, batch_parts):
        # if run on multi-GPUs, this is a list(?)
        if isinstance(batch_parts, list):
            return sum(map(lambda x: x['loss'], batch_parts))
        else:
            return batch_parts['loss']

    def validation_step_end(self, batch_parts):
        if isinstance(batch_parts, list):
            return sum(map(lambda x: x['loss'], batch_parts))
        else:
            return batch_parts['loss']

    def training_epoch_end(self, outputs):
        self.log('Training Accuracy', self.training_accuracy.compute())
        self.log('Training SeqF1', self.training_f1.compute())
        self.training_accuracy.reset()
        self.training_f1.reset()

    def validation_epoch_end(self, outputs):
        self.log('Validation Accuracy', self.validation_accuracy.compute())
        self.log('Validation SeqF1', self.validation_f1.compute())
        self.validation_accuracy.reset()
        self.validation_f1.reset()