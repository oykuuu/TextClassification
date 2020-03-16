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
        """
        Recurrent neural network component of CategoryPredictor. Processes
        the short description sentences.

        Parameters
        ----------
        hidden_dim
            int, size of the hidden layer in LSTM.
        embed_dim
            int, embedding size
        num_linear
            int, number of linear layers
        num_class
            int, number of unique category labels
        lstm_layers
            int, number of LSTM layers

        Returns
        -------
        None
        """
        super(RecurrentNetwork, self).__init__()

        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=lstm_layers, dropout=0.3)
        self.linear_layers = []
        for _ in range(num_linear):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.lstm_linear = nn.Linear(hidden_dim, num_class)

    def forward(self, embed):
        """
        Forward propagation.

        Parameters
        ----------
        embed
            torch.Tensor, embeddings of inputs

        Returns
        -------
        lstm_output
            torch.Tensor, output tensor of recurrent network
            (same size as class numbers)
        """
        hidden, _ = self.encoder(embed)
        feature = hidden[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
            lstm_output = self.lstm_linear(feature)

        return lstm_output


class LabelSimilarity(nn.Module):
    def __init__(self):
        """
        Label similarity component of CategoryPredictor. Calculates
        the cosine similarity between label words and description words.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super(LabelSimilarity, self).__init__()

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-8)

    def forward(self, embed, label_embed):
        """
        Forward propagation.

        Parameters
        ----------
        embed
            torch.Tensor, embeddings of inputs
        label_embed
            dict, dictionary with keys named after label words whose word embeddings
            are stored in the values.
        
        Returns
        -------
        match_output
            torch.Tensor, cosine word similarity scores
            (same size as class numbers)
        """
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
        """
        Integrated Recurrent Network component with Label Similarity
        component to form the CategoryPredictor.

        Parameters
        ----------
        hidden_dim
            int, size of the hidden layer in LSTM.
        vocab_size
            int, vocabulary size used in the word embeddings
        word_embed
            torchtext.vocab, pretrained word embedding if available.
        embed_dim
            int, embedding size
        num_linear
            int, number of linear layers
        num_class
            int, number of unique category labels
        pretrained_embed
            boolean, set to True if using pretrained word embeddings
        lstm_layers
            int, number of LSTM layers

        Returns
        -------
        None
        """
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
        """
        Integrate components for forward propagation.

        Parameters
        ----------
        seq
            torch.Tensor, short description word sequences
        label_embed
            dict, dictionary with keys named after label words whose word embeddings
            are stored in the values.
        
        Returns
        -------
        preds
            torch.Tensor, predicted logits
        """
        # get the word embeddings for short descriptions
        embed = self.embedding(seq)

        # obtain outputs from recurrent network and label similarity model
        lstm_output = self.recurrent(embed)
        similarity_output = self.label_similarity(embed, label_embed)

        # combine outputs and pass through final linear layer
        integrated = torch.cat((lstm_output, similarity_output), axis=1)
        preds = self.predictor(integrated)
        return preds

