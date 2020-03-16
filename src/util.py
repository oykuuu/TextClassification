""" Utility functions used in training the CategoryPredictor.
"""

import os
import numpy as np
import torch

from scipy.spatial.distance import dice
from sklearn.metrics import jaccard_score
from gensim.models import KeyedVectors

from tqdm.notebook import tqdm


def clip_gradient(model, clip_value=0.95):
    """
    Before backpropagation, the gradients are clipped to prevent exploding gradients.

    Parameters
    ----------
    model
        model, CategoryPredictor model in training
    clip_value
        float, number to start clipping at

    Returns
    -------
    None
    """
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def get_test_predictions(
        model, test_dataloader, label_embed, threshold=0.5, metric="jaccard"
):
    """
    Predicts on the test data and returns its evaluation metrics and the predictions.

    Parameters
    ----------
    model
        model, CategoryPredictor model in training
    test_dataloader
        BatchWrapper object, test dataloader to get predictions for
    label_embed
        dict, dictionary with keys named after label words whose word embeddings
        are stored in the values.
    threshold
        float, the threshold above which a logit is classified as 1.
    metric
        string, evaluation metric. ['jaccard', 'dice']

    Returns
    -------
    test_preds
        np.array, N x num_class array of predictions
    avg_score
        float, performance score calculated by using the evalutaion metric
    """
    test_preds = []
    avg_score = 0
    n_test = 0
    for x, y in tqdm(test_dataloader):
        preds = model(x, label_embed)
        preds = preds.data
        preds = torch.sigmoid(preds).data.numpy()
        pred_labels = (preds > threshold) + 0
        test_preds.append(pred_labels)
        if metric == "jaccard":
            score = jaccard_score(y, pred_labels, average="samples")
        elif metric == "dice":
            batch_score = [
                dice(y[i, :], pred_labels[i, :]) for i in range(pred_labels.shape[0])
            ]
            score = np.mean(batch_score)
        avg_score += score
        n_test += x.size(0)
    avg_score /= n_test
    test_preds = np.vstack(test_preds)
    return test_preds, avg_score


def load_fasttext(bin_path="./.vector_cache/wiki.en.vec"):
    """
    Loads the pretrained FastText embeddings using the gensim library.

    Parameters
    ----------
    bin_path
        string, file path to the word2vec formatted FastText bin file.

    Returns
    -------
    fasttext_embed
        gensim.models.KeyedVectors, pretrained FastText word embeddings
    """
    fasttext_embed = KeyedVectors.load_word2vec_format(bin_path, binary=False)
    return fasttext_embed


def label_word_embeddings(label_cols, fasttext_embed):
    """
    Collects word embeddings of all words used in category description.
    Disregards non-descriptive words such as 'and', 'services' in
    category description.

    Parameters
    ----------
    label_cols
        list, list of category labels.
    fasttext_embed
        gensim.models.KeyedVectors, pretrained FastText word embeddings

    Returns
    -------
    label_embed
        dict, dict of label word embeddings
    """
    label_embed = {}
    for label in label_cols:
        words = label.split("_")
        word_embeds = []
        for word in words:
            word = word.lower()
            if word in fasttext_embed and (word != "and" and word != "services"):
                word_embeds.append(
                    torch.from_numpy(fasttext_embed[word]).reshape([1, 1, -1])
                )
        label_embed[label] = word_embeds

    return label_embed
