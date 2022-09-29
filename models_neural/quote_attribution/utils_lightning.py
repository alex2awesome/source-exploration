import pytorch_lightning as pl
from torchmetrics import F1


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

    def training_step(self, batch, batch_idx):
        loss = self.forward(**batch)[0]
        self.log('Training Loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(**batch)[0]
        self.log('Validation loss', loss)
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
        pass

    def validation_epoch_end(self, outputs):
        pass