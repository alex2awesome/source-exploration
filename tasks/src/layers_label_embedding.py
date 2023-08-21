import re

import torch
from torch import nn as nn

from tasks.src.layers_attention import LabelEmbeddingWithContext
from tasks.src.utils_general import get_device


class LabelEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = get_device()
        self.label_embeds = nn.Embedding(self.config.num_output_tags + 2, self.config.hidden_dim)
        if self.config.label_pos_embs:
            self.label_pos_embeds = nn.Embedding(self.config.num_position_embeddings + 1, self.config.hidden_dim)
        self.start_idx = self.config.num_output_tags
        self.end_idx = self.config.num_output_tags + 1
        self.label_attention = LabelEmbeddingWithContext(config=self.config)
        self.s0 = 0 if not self.config.use_headline else 1
        self.s1 = self.s0 + 1

    def get_s_and_e(self):
        if (self.config.num_labels_pred_window == 0
            or (self.config.num_labels_pred_window is None)
            or self.config.separate_heads
        ):
            return 1, 1
        else:
            return 0, 0

    def get_label_idx_range(self, n, head=None):
        if head == 'generate':
            return range(self.s0, n)

        if not self.config.separate_heads:
            if ((self.config.num_labels_pred_window == 0) or (self.config.num_labels_pred_window is None)):
                return range(self.s1, n - 1)
            else:
                return range(self.s0, n)
        else:
            if head == 'main':
                return range(self.s1, n - 1)
            else:
                k = int(re.search('\d', head)[0])
                if 'forward' in head:
                    return list(range(self.s0 + k + 1, n)) + [n - 1] * (k - 1)
                else:
                    return [self.s0] * k + list(range(self.s1, n - k - 1))

    def reformat_labels(self, labels, head=None):
        """labels is padded with (l_s, l_e): i.e.: [l_s, l_0, l_1..., l_n, l_e]"""
        if head == 'generate':
            return labels[self.s0: ]
        if not self.config.separate_heads or head == 'main':
            return labels[self.s1: -1]
        else:
            k = int(re.search('\d', head)[0])
            n = len(labels)
            ignore_tensor = torch.tensor([-100] * (k - 1), device=self.device)
            if 'forward' in head:
                idxs = list(range(self.s0 + k + 1, n))
                output = torch.hstack((labels[idxs], ignore_tensor))
            else:
                idxs = list(range(self.s0, n - k - 1))
                output = torch.hstack((ignore_tensor, labels[idxs]))
            return output.to(int)

    def forward(self, labels, head=None):
        if isinstance(labels, list):
            labels = [self.start_idx] + labels + [self.end_idx]
            labels = torch.tensor(labels, device=self.device)
        else:
            start_lab = torch.tensor([self.start_idx], device=self.device)
            end_lab = torch.tensor([self.end_idx], device=self.device)
            labels = torch.hstack((start_lab, labels, end_lab))

        label_embedding_mat = self.label_embeds(labels)
        if self.config.label_pos_embs:
            position_ids = torch.arange(len(labels), dtype=torch.long, device=self.device)
            position_ids = position_ids.where(position_ids < self.config.num_position_embeddings, torch.tensor(self.config.num_position_embeddings, device=self.device))
            pos_emb = self.label_pos_embeds(position_ids)
            label_embedding_mat = label_embedding_mat + pos_emb

        output_label_embs = []
        to_iterate = self.get_label_idx_range(len(label_embedding_mat), head)
        for label_idx in to_iterate:
            windowed_embedding = self.label_attention(label_embedding_mat, label_idx)
            output_label_embs.append(windowed_embedding)
        output_label_embs = torch.vstack(output_label_embs)
        labels = self.reformat_labels(labels, head)
        return output_label_embs, labels