import torch


class BasicAccuracy():
    def __init__(self, num_window_tokens=2):
        self.internal_list = []
        self.w = num_window_tokens

    def calculate_overlap(self, start_label, end_label, start_logits, end_logits):
        """Calculates the match of a single example using the following heuristic:
            * There must be at least one token in the true source predicted.
            * The start/end points cannot be more than `w` tokens away from the true start/end points.
        """
        start_pred = start_logits.argmax()
        end_pred = end_logits.argmax()
        start_pred, end_pred = min(start_pred, end_pred), max(start_pred, end_pred)

        has_overlap = (start_pred <= start_label and end_pred >= start_label) or \
                      (start_pred <= start_label and end_pred >= end_label)
        two_tokens = (abs(start_pred - start_label) <= self.w) and (abs(end_pred - end_label))
        return has_overlap and two_tokens

    def __call__(self, preds, labels):
        start_logits, end_logits = preds
        start_label, end_label = labels

        pred = self.calculate_overlap(start_label, end_label, start_logits, end_logits)

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


class SequenceF1():
    def __init__(self):
        self.internal_list = []

    def calculate_f1(self, start_label, end_label, start_logits, end_logits):
        start_pred = start_logits.argmax()
        end_pred = end_logits.argmax()
        start_pred, end_pred = min(start_pred, end_pred), max(start_pred, end_pred)

        common_start = max(start_pred, start_label)
        common_end = min(end_pred, end_label)
        common_tokens = common_end - common_start

        len_pred_tokens = end_pred - start_pred
        len_truth_tokens = end_label - start_label

        prec = len(common_tokens) / len_pred_tokens
        rec = len(common_tokens) / len_truth_tokens

        return 2 * (prec * rec) / (prec + rec)

    def __call__(self, preds, labels):
        start_logits, end_logits = preds
        start_label, end_label = labels

        f1 = self.calculate_f1(start_label, end_label, start_logits, end_logits)
        self.internal_list.append(f1)

    def compute(self):
        if len(self.internal_list) == 0:
            return 0
        return sum(self.internal_list) / len(self.internal_list)

    def reset(self):
        self.internal_list = []
