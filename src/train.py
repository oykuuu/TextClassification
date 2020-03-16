""" Trains the CategoryPredictor and writes the best model checkpoints and
the final trained model into an output directory.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, FastText
from torchtext.data import Iterator, BucketIterator
from sklearn.metrics import jaccard_score


import pandas as pd
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

# import tqdm
from tqdm.notebook import tqdm


from data.dataloader import preprocess, get_nlp_dataset, get_nlp_dataloader
from model.models import CategoryPredictor
from util import (
    clip_gradient,
    get_test_predictions,
    load_fasttext,
    label_word_embeddings,
)


def train_epoch(
    model,
    train_dataloader,
    valid_dataloader,
    label_embed,
    epochs=3,
    learning_rate=1e-3,
    clip=0.95,
    threshold=0.5,
    output_dir="./",
):
    """
    Predicts on the test data and returns its evaluation metrics and the predictions.

    Parameters
    ----------
    model
        nn.Module, CategoryPredictor model in training
    train_dataloader
        BatchWrapper object, train dataloader used in training
    valid_dataloader
        BatchWrapper object, validation dataloader used in evaluation
    label_embed
        dict, dictionary with keys named after label words whose word embeddings
        are stored in the values.
    epochs
        int, number of training epochs
    learning rate
        float, initial learning rate. (will be updated by Adam optimizer)
    clip
        float, gradient clipping threshold
    threshold
        float, the threshold above which a logit is classified as 1.
    output_dir
        string, folder to save trained models

    Returns
    -------
    model
        nn.Module, final trained CategoryPredictor model
    best_model
        nn.Module, best trained CategoryPredictor model
    """
    try:
        learning_rate = 1e-3
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss(reduction="sum")

        count = 0
        best_jaccard = -99
        best_model = None
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            running_corrects = 0
            model.train()
            n_train = 0
            for x, y in tqdm(train_dataloader):
                opt.zero_grad()

                preds = model(x, label_embed)
                loss = criterion(preds, y)
                loss.backward()
                clip_gradient(model, 0.95)
                opt.step()

                n_train += x.size(0)
                running_loss += loss.data * x.size(0)

                if count % 500 == 0:
                    print("{} - loss: {}".format(count, loss.data))
                count += 1
            epoch_loss = running_loss / len(train_dataloader)

            # calculate the validation loss for this epoch
            val_loss = 0.0
            model.eval()  # turn on evaluation mode
            for x, y in valid_dataloader:
                preds = model(x, label_embed)
                loss = criterion(y, preds)
                val_loss += loss.data * x.size(0)

            val_loss /= len(valid_dataloader)

            valid_preds, avg_jaccard = get_test_predictions(
                model, valid_dataloader, label_embed, threshold=0.5
            )
            if avg_jaccard > best_jaccard:
                best_jaccard = avg_jaccard
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"model_epoch_{epoch}.pkl"),
                )
                best_model = model

            print(
                "Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Jaccard Score: {:.4f}".format(
                    epoch, epoch_loss, val_loss, avg_jaccard
                )
            )
        torch.save(
            model.state_dict(), os.path.join(output_dir, "model_epoch_final.pkl")
        )

    except KeyboardInterrupt:

        print("Saving model before interupting")
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"model_epoch_{epoch}.interrupt.pkl"),
        )
    except KeyboardInterrupt:

        print("Saving model before interupting")
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"model_epoch_{epoch}.interrupt.pkl"),
        )

    return model, best_model


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    folder = "~/Desktop/Oyku/Vela/org_data"
    output_dir = os.path.join(folder, "output")
    id_col = "id"
    text_col = "text"
    train_fname = "small_train.csv"
    val_fname = "small_val.csv"
    test_fname = "small_val.csv"
    batch_size = 64
    output_dir = "~/Desktop/Oyku/Vela/org_data/output/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Output directory created in: {}".format(output_dir))

    train_df = pd.read_csv(os.path.join(folder, train_fname))
    label_cols = [
        col
        for col in train_df.columns
        if not (col == "Unnamed: 0") and not (col == id_col) and not (col == text_col)
    ]

    fields, TEXT, LABEL = preprocess(
        train_df, id_col=id_col, text_col=text_col, label_cols=label_cols
    )
    train_dataset, valid_dataset, test_dataset, TEXT, LABEL = get_nlp_dataset(
        fields,
        TEXT,
        LABEL,
        vocab="FastText",
        folder=folder,
        train_fname=train_fname,
        val_fname=val_fname,
        test_fname=test_fname,
    )
    train_dataloader, valid_dataloader, test_dataloader = get_nlp_dataloader(
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_size,
        text_col,
        label_cols,
    )

    bin_path = "./.vector_cache/wiki.en.vec"
    fasttext_embed = load_fasttext(bin_path)
    label_embed = label_word_embeddings(label_cols, fasttext_embed)

    hidden_dim = 128
    embed_dim = 300
    num_linear = 1
    vocab_size = len(TEXT.vocab)
    num_class = len(label_cols)
    pretrained = True
    word_embed = TEXT.vocab.vectors
    epochs = 1
    learning_rate = 1e-3
    clip = 0.95
    threshold = 0.5

    # model = SimpleLSTMBaseline(hidden_dim, vocab_size, word_embed, embed_dim, num_linear, num_class, pretrained)
    model = CategoryPredictor(
        hidden_dim,
        vocab_size,
        word_embed,
        embed_dim,
        num_linear,
        num_class,
        pretrained,
        lstm_layers=2,
    )
    final_model, best_model = train_epoch(
        model,
        train_dataloader,
        valid_dataloader,
        label_embed,
        epochs,
        learning_rate,
        clip,
        threshold,
        output_dir,
    )
    test_preds, jaccard_index = get_test_predictions(
        best_model, test_dataloader, label_embed, threshold
    )

    return final_model, best_model, jaccard_index


if __name__ == "__main__":

    final_model, best_model, jaccard_index = main()
