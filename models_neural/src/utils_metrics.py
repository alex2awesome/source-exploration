from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
from torch.distributions import Categorical
import torch
from torchmetrics import Metric
import transformers
from packaging import version
from torch import nn


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
    if not getattr(config, 'do_multitask', False):
        pred_window = getattr(config, 'num_labels_pred_window', None)
        if ((pred_window is not None) and (pred_window != 0)):
            if getattr(config, 'separate_heads', False):
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


def _get_words(word_ids, start_idx, end_idx):
    if len(word_ids.size()) > 1:
        return word_ids[:, start_idx:end_idx]
    return word_ids[start_idx:end_idx].clone()


def _set_words(word_ids, start_idx, end_idx, val):
    if len(word_ids.size()) > 1:
        word_ids[:, start_idx:end_idx] = val
    word_ids[start_idx:end_idx] = val
    return word_ids.clone()


class PerplexityBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def calculate_ppl_for_sequence(self, input_ids, labels, n_words_in_input, model):
        start_idx, end_idx = 0, 0
        for word_idx in range(0, n_words_in_input, self.stride):
            start_idx = max(word_idx + self.stride - self.max_context_len, 0)
            end_idx = min(word_idx + self.stride, n_words_in_input)
            target_len = end_idx - word_idx

            input_t = _get_words(input_ids, start_idx, end_idx)
            output_t = _get_words(labels, start_idx, end_idx)
            output_t = _set_words(output_t, 0, -target_len, -100)

            with torch.no_grad(): # to make sure there's no gradient computation
                if version.parse(transformers.__version__) > version.parse('4.0.0'):
                    loss = model.forward(input_ids=input_t, labels=output_t, return_dict=False)[0]
                else:
                    loss = model.forward(input_ids=input_t, labels=output_t)[0]
                loss = loss * target_len
                loss = loss.to(self.device)

            self.lls += loss
        #
        self.count += torch.tensor(end_idx, device=self.device)

    def update(self, input_ids, model):
        labels = input_ids.clone()
        if False and len(labels.size()) > 1: # then, first dim is batch (disabling this for now... for RoBERTa, we actually need a 2-d input)
            for input_i, labels_i in zip(input_ids, labels):
                n_words_in_input = input_i.size()[0]
                self.calculate_ppl_for_sequence(input_i, labels_i, n_words_in_input, model)
        else:
            n_words_in_input = labels.size()[0]
            self.calculate_ppl_for_sequence(input_ids, labels, n_words_in_input, model)

    def reset(self):
        self.lls = torch.tensor(0.0, device=self.device)
        self.count = torch.tensor(0.0, device=self.device)

    def compute(self):
        return torch.exp(self.lls / self.count)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            self.update(*args, **kwargs)


class PerplexityMetric(Metric, PerplexityBase):
    def __init__(self, device, stride=128, max_context_len=2048, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.device = device
        self.add_state('lls', default=torch.tensor(0.0, device=self.device), dist_reduce_fx="sum")
        self.add_state('count', default=torch.tensor(0.0, device=self.device), dist_reduce_fx='sum')
        self.stride = stride
        self.max_context_len = max_context_len


class PerplexityOrig(PerplexityBase):
    def __init__(self, device, stride=128, max_context_len=2048):
        super().__init__()
        self.device = device
        self.lls = torch.tensor(0.0, device=self.device)
        self.count = torch.tensor(0.0, device=self.device)
        self.stride = stride
        self.max_context_len = max_context_len
