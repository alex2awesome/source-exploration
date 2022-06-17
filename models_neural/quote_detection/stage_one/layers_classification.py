import torch
from torch import nn

from .utils_general import get_config, get_device, _get_attention_mask
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100

def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output


class MultiClassMixin(nn.Module):
    """
    Takes in a tensor of sentence embeddings and a document embedding. Concatenates and performs a classification.
    """

    def __init__(self, *args, **kwargs):#  config, num_output_tags):
        super().__init__(*args, **kwargs)
        # final classification head
        # accounts for 1 of them.
        self.hidden_dim = self.config.hidden_dim
        self.config = get_config(kwargs=kwargs)
        self.use_tsa = self.config.use_tsa
        self.init_pred_layers()
        self.drop = nn.Dropout(self.config.dropout)
        self._init_classifier_prediction_weights()
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    def init_pred_layers(self):
        if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
            self.text_and_label_comb = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        if self.config.separate_heads:
            self.pred = nn.Linear(self.hidden_dim, self.config.num_output_tags + 2)
        else:
            self.pred = nn.Linear(self.hidden_dim, self.config.num_output_tags)


    def _init_classifier_prediction_weights(self):
        nn.init.xavier_uniform_(self.pred.state_dict()['weight'])
        self.pred.bias.data.fill_(0)
        if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
            nn.init.xavier_uniform_(self.text_and_label_comb.state_dict()['weight'])
            self.text_and_label_comb.bias.data.fill_(0)

    def calculate_loss(self, preds, labels):
        if len(labels.shape) == 0:
            labels = labels.unsqueeze(dim=0)
        # if len(labels) == 1:  # only interested in the prediction on the last sentence.
        #     preds = preds[[-1]]
        loss = self.criterion(preds, labels)
        return loss

    def tsa(self, loss, logits, label_ids, global_step):
        """
        Method in Unsupervised Data Augmentation designed to implement a form of curriculum learning.
        """
        tsa_thresh = get_tsa_thresh(
            self.config.tsa_schedule,
            global_step,
            self.config.total_steps,
            start=1. / logits.shape[-1],
            end=1
        )
        tsa_thresh = tsa_thresh.to(device=loss.device)
        larger_than_threshold = torch.exp(-loss) > tsa_thresh
        loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        loss = torch.sum(loss * loss_mask, dim=-1)
        loss = loss / torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1., device=loss.device))
        return loss

    def classification(
            self,
            hidden_embs,
            labels=None,
            label_embs=None,
            get_loss=True,
            global_step=None
    ):
        # loss
        if (self.config.label_context_back != 0) or (self.config.label_context_forward != 0):
            label_embs = label_embs.reshape(hidden_embs.shape)
            hidden_embs = torch.hstack((hidden_embs, label_embs))
            hidden_embs = self.text_and_label_comb(self.drop(torch.tanh(hidden_embs)))

        prediction = self.pred(self.drop(torch.tanh(hidden_embs)))  # pred = ( batch_size x num_labels)
        if (get_loss == False) or (labels is None):
            return None, prediction
        #
        loss = self.calculate_loss(prediction, labels)
        if self.use_tsa and (global_step is not None):
            loss = self.tsa(loss, prediction, labels, global_step=int(global_step / 2))
        else:
            loss = torch.mean(loss)
        return loss, prediction, labels


class MultiTaskMultiClassMixin(MultiClassMixin):
    """
    Takes in a tensor of sentence embeddings and a document embedding. Concatenates and performs a classification.
    """

    def __init__(self, *args, **kwargs):#  config, num_output_tags):
        self.device = get_device()
        super().__init__(*args, **kwargs)

    def get_combiner(self):
        downsize_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        act = nn.Tanh()
        return nn.Sequential(downsize_layer, act)

    def get_pred_module(self, n):
        pre_pred = nn.Linear(self.hidden_dim, self.hidden_dim)
        act = nn.Tanh()
        pred = nn.Linear(self.hidden_dim, n)
        return nn.Sequential(pre_pred, act, pred)

    def init_pred_layers(self):
        # generic multitask
        if self.config.num_labels_pred_window == 0:
            self.preds = nn.ModuleList()
            self.num_tasks = self.config.num_output_tags
            for num_output_tags_task in self.config.num_output_tags:
                pred = self.get_pred_module(num_output_tags_task)
                self.preds.append(pred)
            self.loss_weighting = torch.tensor(self.config.loss_weighting, device=self.device)

        # multitask for label look-aheads
        else:
            self.preds = nn.ModuleDict()
            self.combiners = nn.ModuleDict()
            self.num_tasks = (self.config.num_labels_pred_window * 2) + 1
            self.start_idx = self.config.num_output_tags
            self.end_idx = self.config.num_output_tags + 1

            # get tasks
            # main task
            self.preds['main'] = self.get_pred_module(self.config.num_output_tags + 2)
            self.combiners['main'] = self.get_combiner()

            # tasks before
            for i in range(self.config.num_labels_pred_window):
                self.preds['backwards %s' % (i + 1)] = self.get_pred_module(self.config.num_output_tags + 2)
                self.combiners['backwards %s' % (i + 1)] = self.get_combiner()

            # tasks afterwards
            for i in range(self.config.num_labels_pred_window):
                self.preds['forward %s' % (i + 1)] = self.get_pred_module(self.config.num_output_tags + 2)
                self.combiners['forward %s' % (i + 1)] = self.get_combiner()

            # loss-weighting (needs to change)
            alpha = self.config.loss_weighting[0] if isinstance(self.config.loss_weighting, list) else self.config.loss_weighting
            alpha_vec = [alpha ** x for x in range(self.num_tasks)]
            self.loss_weighting = torch.tensor(alpha_vec, device=self.device, requires_grad=True)

    def _init_classifier_prediction_weights(self):
        if isinstance(self.preds, nn.ModuleList):
            for pred in self.preds:
                nn.init.xavier_uniform_(pred.state_dict()['weight'])
                pred.bias.data.fill_(0)

    def calculate_loss(self, preds, labels):
        if self.config.num_labels_pred_window == 0:
            losses = torch.zeros(self.num_tasks, device=self.device)
            for task_idx, (task_pred, task_label) in enumerate(zip(preds, labels)):
                loss = self.criterion(task_pred, task_label)
                if not self.config.do_doc_pred:
                    loss = loss.sum()
                losses[task_idx] = loss
        else:
            losses = {}
            for k in preds.keys():
                loss = self.criterion(preds[k], labels[k])
                if not self.config.do_doc_pred:
                    loss = loss.sum()
                losses[k] = loss
            losses = torch.tensor(list(losses.values()), device=self.device)

        return self.loss_weighting.dot(losses)

    def _combine(self, h, l, layer):
        h = torch.hstack((h, l))
        return layer(self.drop(torch.tanh(h)))

    def _format_x(self, hidden_embs, label_embs):
        # label embs is of size (num_labels + 2)
        if label_embs is not None:
            x = {}
            # main
            x['main'] = self._combine(hidden_embs, label_embs[1:-1], self.combiners['main'])

            # backwards
            for i in range(1, self.local_pred_window + 1):
                l_s = torch.vstack([label_embs[0]] * i)
                l = torch.vstack((l_s, label_embs[1:-(i+1)]))
                x['backwards %s' % i] = self._combine(hidden_embs, l, self.combiners['backwards %s' % i])

            # forwards
            for i in range(1, self.local_pred_window + 1):
                l_e = torch.vstack([label_embs[-1]] * i)
                l = torch.vstack((label_embs[i:-2], l_e))
                x['forward %s' % i] = self._combine(hidden_embs, l, self.combiners['forward %s' % i])
            return x
        else:
            return hidden_embs

    def _format_y(self, labels):
        """
            Returns an output grid of labels for the multitask prediction problem.
            Returns list of len `num_labels_ahead`, where each item is a tensor of len `len(labels)`.
                * For output[0], this is the original label vector (of length: num_sentences)
                * For output[i], this is the label vector shifted by `i` with one special end
                    `label` at the end, and i-1 `ignore` label padding.
        """
        if self.config.num_labels_pred_window == 0:
            return labels.T
        else:
            all_labels = {}

            # main
            all_labels['main'] = labels

            # backwards
            for i in range(1, self.local_pred_window + 1):
                start_tok = torch.tensor([self.start_idx], device=self.device)
                ignore_toks = torch.tensor([IGNORE_INDEX] * (i - 1), device=self.device)
                labels_i = torch.hstack((ignore_toks, start_tok, labels[:-i])).to(int)
                all_labels['backwards %s' % i] = labels_i

            # forward
            for i in range(1, self.local_pred_window + 1):
                end_tok = torch.tensor([self.end_idx], device=self.device)
                ignore_toks = torch.tensor([IGNORE_INDEX] * (i - 1), device=self.device)
                labels_i = torch.hstack((labels[i:], end_tok, ignore_toks)).to(int)
                all_labels['forward %s' % i] = labels_i

            # return
            return all_labels

    def _get_probas(self, hidden_embs):
        if self.config.num_labels_pred_window == 0:
            preds = []
            for task_idx in range(self.num_tasks):
                pred = self.preds[task_idx](hidden_embs)  # pred = ( batch_size x num_labels)
                preds.append(pred)
            return preds
        else:
            preds = {}
            for k in hidden_embs.keys():
                preds[k] = self.preds[k](hidden_embs[k])
            return preds

    def _get_preds_from_probas(self, preds):
        if isinstance(preds, list):
            preds = list(map(lambda x: x.squeeze(), preds))
            if self.config.do_doc_pred:
                preds = pad_sequence(preds, batch_first=True)
                a = _get_attention_mask(preds, max_length_seq=100).to(self.device)
                preds = preds + ((a - 1) * 1000)
            else:
                preds = list(map(lambda x: x.argmax(dim=1), preds))
                preds = torch.vstack(preds).T
            return preds
        else:
            output_preds = {}
            for k, v in preds.items():
                output_preds[k] = v.argmax(dim=1)
            return output_preds

    def classification(
            self,
            hidden_embs,
            labels=None,
            label_embs=None,
            get_loss=True,
            global_step=None
    ):
        self.local_pred_window = min(len(hidden_embs), self.config.num_labels_pred_window)

        # loss
        hidden_embs = self.drop(torch.tanh(hidden_embs))
        hidden_embs = self._format_x(hidden_embs, label_embs)
        labels = self._format_y(labels)
        preds = self._get_probas(hidden_embs)
        if (get_loss == False) or (labels is None):
            return None, preds
        #
        loss = self.calculate_loss(preds, labels)
        loss = torch.mean(loss)
        preds = self._get_preds_from_probas(preds)
        return loss, preds, labels
