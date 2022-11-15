import pytorch_lightning as pl
from torchmetrics import F1, Accuracy


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


import torch
class BasicAccuracy():
    def __init__(self):
        self.internal_list = []

    def __call__(self, pred):
        if isinstance(pred, torch.Tensor):
            if len(pred.shape) > 0:
                pred = pred[0]
                pred = float(pred.to('cpu'))
        self.internal_list.append(pred )

    def compute(self):
        if len(self.internal_list) == 0:
            return 0
        return sum(self.internal_list) / len(self.internal_list)

    def reset(self):
        self.internal_list = []

class LightningQASteps(pl.LightningModule):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.training_accuracy = BasicAccuracy()
        self.validation_accuracy = BasicAccuracy()

    def calculate_overlap(self, start_label, end_label, start_logits, end_logits):
        """Calculates the match of a single example using the following heuristic:
            * There must be at least one token in the true source predicted.
            * The start/end points cannot be more than 2 tokens away from the true start/end points.
        """
        start_pred = start_logits.argmax()
        end_pred = end_logits.argmax()
        start_pred, end_pred = min(start_pred, end_pred), max(start_pred, end_pred)

        has_overlap = (start_pred <= start_label and end_pred >= start_label) or \
                      (start_pred <= start_label and end_pred >= end_label)
        two_tokens = (abs(start_pred - start_label) <= 2) and (abs(end_pred - end_label))
        return has_overlap and two_tokens

    def handle_output(self, outputs, batch):
        loss, start_logits, end_logits = outputs[0], outputs[1], outputs[2]
        start_label, end_label = batch['start_positions'], batch['end_positions']
        overlap = self.calculate_overlap(start_label, end_label, start_logits, end_logits)
        return loss, overlap

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss, accuracy = self.handle_output(output, batch)
        self.log('Training Loss', loss)
        self.training_accuracy(accuracy)
        return {'loss': loss, 'is_overlap': accuracy}

    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss, accuracy = self.handle_output(output, batch)
        self.log('Validation loss', loss)
        self.validation_accuracy(accuracy)
        return {'loss': loss, 'is_overlap': accuracy}

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
        acc = self.validation_accuracy.compute()
        self.log('Training Accuracy', acc)
        self.validation_accuracy.reset()

    def validation_epoch_end(self, outputs):
        acc = self.validation_accuracy.compute()
        self.log('Validation Accuracy', acc)
        self.validation_accuracy.reset()