import torch
from torchmetrics import Metric, MetricCollection, Accuracy, Precision, Recall, F1
from torch.distributions import Categorical

class ClassCountsBase(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.class_name_pattern = 'class_num_%s'
        for i in range(self.num_classes):
            self.add_state(self.class_name_pattern % i, default=torch.tensor(0), dist_reduce_fx="sum")

    def _get_class_name(self, i):
        if not isinstance(i, (float, int)):
            if isinstance(i, type(torch.tensor(1))):
                i = int(i.detach().cpu().numpy())
            else:
                raise ValueError('Unknown type: %s' % i)
        return self.class_name_pattern % i

    def _get_class_counts(self):
        output = []
        for i in range(self.num_classes):
            output.append(getattr(self, self._get_class_name(i)))
        return output

    def update(self, preds):
        for pred in preds:
            curr_count = getattr(self, self._get_class_name(pred))
            setattr(self, self._get_class_name(pred), curr_count + 1)

    def reset(self):
        for i in range(self.num_classes):
            setattr(self, self._get_class_name(i), torch.tensor(0))


class Entropy(ClassCountsBase):
    def compute(self):
        class_counts = torch.tensor(self._get_class_counts())
        return Categorical(probs=class_counts).entropy()


class MaxCount(ClassCountsBase):
    def compute(self):
        class_counts = self._get_class_counts()
        return max(class_counts)

from torch import nn
class MultitaskReport(nn.Module):
    def __init__(self, num_classes, do_doc, dist_sync_on_step):
        super().__init__()
        self.do_doc = do_doc
        self.task_metrics = nn.ModuleList()
        for task_idx, n in enumerate(range(num_classes)):
            m = _get_classification_report_metric(n, dist_sync_on_step)
            self.task_metrics.append(m)

    def __call__(self, y_pred, y_true):
        # batch['y_pred'], batch['y_true']
        y_true = y_true.T
        if self.do_doc:
            y_pred = y_pred.unsqueeze(dim=0)
        y_pred = y_pred.T
        for task_idx, (pred, true) in enumerate(zip(y_pred, y_true)):
            self.task_metrics[task_idx](pred, true)

    def compute(self):
        output = []
        for t in self.task_metrics:
            c = t.compute()
            output.append(c)
        return output

    def reset(self):
        for t in self.task_metrics:
            t.reset()

    def to(self, device):
        self.task_metrics.to(device)


class StructuredPredReport(nn.Module):
    def __init__(self, config, dist_sync_on_step):
        super().__init__()
        n = config.num_output_tags + 2
        self.task_metrics = nn.ModuleDict()
        # main
        self.task_metrics['main'] = _get_classification_report_metric(n, dist_sync_on_step)
        # backwards
        for i in range(config.num_labels_pred_window):
            m = _get_classification_report_metric(n, dist_sync_on_step)
            self.task_metrics['backwards %s' % (i + 1)] = m
        # forwards
        for i in range(config.num_labels_pred_window):
            m = _get_classification_report_metric(n, dist_sync_on_step)
            self.task_metrics['forward %s' % (i + 1)] = m

    def __call__(self, y_pred, y_true, *args, **kwargs):
        # batch['y_pred'], batch['y_true']
        for y_pred_i, y_true_i in zip(y_pred, y_true):
            for k, m in self.task_metrics.items():
                y_t = y_true_i[k]
                y_p = y_pred_i[k]
                to_keep = torch.where(y_t != -100)
                m(y_p[to_keep], y_t[to_keep])

    def compute(self):
        output = {}
        for k, t in self.task_metrics.items():
            c = t.compute()
            output[k] = c
        return output

    def reset(self):
        for k, t in self.task_metrics.items():
            t.reset()

    def to(self, device):
        self.task_metrics.to(device)


from collections import defaultdict
class StructuredPredReportSepHeads(StructuredPredReport):
    def __init__(self, config, dist_sync_on_step):
        super().__init__(config, dist_sync_on_step)
        self.updated = defaultdict(int)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        head = kwargs.get('head')
        to_keep = torch.where(y_true != -100)
        for y_p_i, head_i in zip(y_pred, head):
            self.updated[head_i] += 1
            self.task_metrics[head_i](y_p_i[to_keep], y_true[to_keep])

    def compute(self):
        output = {}
        for k, t in self.task_metrics.items():
            if self.updated[k] > 0:
                output[k] = t.compute()
        return output



def _get_classification_report_metric(num_classes, dist_sync_on_step):
     return MetricCollection({
        'Accuracy': Accuracy(),
        'Class %s Precision': Precision(num_classes=num_classes, average=None, dist_sync_on_step=dist_sync_on_step),
        'Class %s Recall': Recall(num_classes=num_classes, average=None, dist_sync_on_step=dist_sync_on_step),
        'Class %s F1': F1(num_classes=num_classes, average=None, dist_sync_on_step=dist_sync_on_step),
        'Weighted Precision': Precision(num_classes=num_classes, average='weighted', dist_sync_on_step=dist_sync_on_step),
        'Weighted Recall': Recall(num_classes=num_classes, average='weighted', dist_sync_on_step=dist_sync_on_step),
        'Weighted F1': F1(num_classes=num_classes, average='weighted', dist_sync_on_step=dist_sync_on_step),
        'Macro Precision': Precision(num_classes=num_classes, average='macro', dist_sync_on_step=dist_sync_on_step),
        'Marco Recall': Recall(num_classes=num_classes, average='macro', dist_sync_on_step=dist_sync_on_step),
        'Macro F1': F1(num_classes=num_classes, average='macro', dist_sync_on_step=dist_sync_on_step)
    })

def get_classification_report_metric(config, dist_sync_on_step):
    if not config.do_multitask:
        if ((config.num_labels_pred_window is not None) and (config.num_labels_pred_window != 0)):
            if config.separate_heads:
                return StructuredPredReportSepHeads(config, dist_sync_on_step)
            else:
                return StructuredPredReport(config, dist_sync_on_step)
        else:
            return _get_classification_report_metric(config.num_output_tags, dist_sync_on_step)
    else:
        return MultitaskReport(
            config.num_output_tags,
            config.do_doc_pred,
            dist_sync_on_step
        )


def format_classification_report(report, step, config):
    """
    Flatten classification report (which includes vectors of len `num_classes`).
    """
    output = {}
    if not (config.do_multitask or
    (config.num_labels_pred_window is not None and
     config.num_labels_pred_window != 0)):
        for k, v in report.items():
            if len(v.shape) == 0:
                output[step + ':' + k] = v
            else:
                for class_idx, v_i in enumerate(v):
                    output[step + ':' + k % class_idx] = v_i
    else:
        to_iterate = enumerate(report) if isinstance(report, list) else report.items()
        for task_idx, task_report in to_iterate:
            for k, v in task_report.items():
                if len(v.shape) == 0:
                    output['Task %s, %s: %s' % (task_idx, step, k)] = v
                else:
                    for class_idx, v_i in enumerate(v):
                        k_i = k % class_idx
                        output['Task %s, %s: %s' % (task_idx, step, k_i)] = v_i
    return output