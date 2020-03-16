""" Models used in Category Prediction.
"""

import torch
import torch.nn as nn

class RecurrentNetwork(nn.Module):
    def __init__(
            self,
            hidden_dim,
            emb_dim=300,
            num_linear=1,
            num_class=46,
            lstm_layers=2,
    ):
        super(RecurrentNetwork, self).__init__()

        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=lstm_layers, dropout=0.3)
        self.linear_layers = []
        for _ in range(num_linear):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.lstm_linear = nn.Linear(hidden_dim, num_class)

    def forward(self, embed):
        hidden, _ = self.encoder(embed)
        feature = hidden[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
            lstm_output = self.lstm_linear(feature)

        return lstm_output


class LabelSimilarity(nn.Module):
    def __init__(self):
        super(LabelSimilarity, self).__init__()

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-8)

    def forward(self, embed, label_embed):
        match = []
        batch_size = embed.size(1)
        for key, value_list in label_embed.items():
            match_score = -1 * torch.ones([batch_size])
            for value in value_list:
                word_score = torch.max(self.cos(embed, value), axis=0).values
                concat = torch.cat(
                    (word_score.reshape([1, -1]), match_score.reshape([1, -1]))
                )
                match_score = torch.max(concat, axis=0).values
            match.append(match_score.reshape([-1, 1]))
        match_output = torch.cat(match, axis=1)

        return match_output


class CategoryPredictor(nn.Module):
    def __init__(
            self,
            hidden_dim,
            vocab_size,
            word_embed=None,
            emb_dim=300,
            num_linear=1,
            num_class=46,
            pretrained_embed=True,
            lstm_layers=2,
    ):
        super(CategoryPredictor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if pretrained_embed:
            self.embedding.weight = nn.Parameter(word_embed, requires_grad=False)
        self.recurrent = RecurrentNetwork(
            hidden_dim,
            emb_dim,
            num_linear,
            num_class,
            lstm_layers,
        )
        self.label_similarity = LabelSimilarity()
        self.predictor = nn.Linear(num_class * 2, num_class)

    def forward(self, seq, label_embed):
        # get the word embeddings for short descriptions
        embed = self.embedding(seq)

        # obtain outputs from recurrent network and label similarity model
        lstm_output = self.recurrent(embed)
        similarity_output = self.label_similarity(embed, label_embed)

        # combine outputs and pass through final linear layer
        integrated = torch.cat((lstm_output, similarity_output), axis=1)
        preds = self.predictor(integrated)
        return preds

