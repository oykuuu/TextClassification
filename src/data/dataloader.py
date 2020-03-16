""" Loads train, validation and test datasets and stores them in dataloaders.
"""
import os
import re
import string
import pandas as pd

from torchtext import data
from torchtext.vocab import GloVe, FastText
from torchtext.data import Iterator, BucketIterator


class BatchWrapper():
    def __init__(self, dataloader, x_var, y_vars):
        """
        Initializes wrapper for the dataloaders for convenient iteration over the data.

        Parameters
        ----------
        dataloader
            torchtext.data.BucketIterator, iterator over a dataset.
        x_var
            string, name of the input text column
        y_vars
            list of strings, names of the label columns in a list

        Returns
        -------
        None
        """
        self.dataloader = dataloader
        self.x_var = x_var
        self.y_vars = y_vars

    def __iter__(self):
        """
        Defines iteration process.

        Parameters
        ----------
        None

        Returns
        -------
        data
            tuple, tuple of data and label returned
        """
        for batch in self.dataloader:
            x = getattr(batch, self.x_var)
            if not (self.y_vars is None):
                y = torch.cat(
                    [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1
                ).float()
            else:
                y = None
            data = (x, y)
            yield data

    def __len__(self):
        """
        Returns length of dataloader.
        """
        return len(self.dataloader)


def preprocess(train_df, id_col="id", text_col="text", label_cols=[]):
    """
    Processes the inputs and the labels. Tokenizes the inputs by extracting individual
    words from the sentence. Also assigns LABEL or TEXT Field objects to columns.

    Parameters
    ----------
    train_df
        pandas.DataFrame, training data.
    id_col
        string, name of the id column
    text_col
        string, name of the input text column
    label_cols
        list of strings, names of the label columns in a list

    Returns
    -------
    fields
        list of tuples, an entry is a tuple of column name and its Field object
    TEXT
        torchtext.data.Field, defines how to convert a text column data into Tensor
    LABEL
        torchtext.data.Field, defines how to convert label columns data into Tensor
    """
    #tokenize = lambda x: x.split()
    tokenize = lambda x: re.findall(r'\w+', x)
    # inputs are sequential as the order of the words are important in a sentence.
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
    # label column has two values: 0, 1. So it isn't sequential and it doesn't
    # have a vocabulary.
    LABEL = data.Field(sequential=False, use_vocab=False)

    fields = []
    for col in train_df.columns:
        if col == id_col or col == "Unnamed: 0":
            fields.append((col, None))
        elif col == text_col:
            fields.append((col, TEXT))
        elif col in label_cols:
            fields.append((col, LABEL))
        else:
            fields.append((col, None))
            print("Warning: {} column is not a text, label or id column.".format(col))

    return fields, TEXT, LABEL


def get_nlp_dataset(
        fields,
        TEXT,
        LABEL,
        vocab="FastText",
        folder="./",
        train_fname="train.csv",
        val_fname="val.csv",
        test_fname=None,
):
    """
    Builds library with a given vocabulary. Forms train, validation and test datasets.

    Parameters
    ----------
    fields
        list of tuples, an entry is a tuple of column name and its Field object
    TEXT
        torchtext.data.Field, defines how to convert a text column data into Tensor
    LABEL
        torchtext.data.Field, defines how to convert label columns data into Tensor
    vocab
        string, pretrained word embedding vocabulary name. ['FastText', 'GloVe']
    folder
        string, folder name where datasets are stored
    train_fname
        string, train data filename
    val_fname
        string, validation data filename
    test_fname
        string, test data filename

    Returns
    -------
    train_dataset
        torchtext.data.TabularDataset, dataset object for training
    valid_dataset
        torchtext.data.TabularDataset, dataset object for validation
    test_dataset
        torchtext.data.TabularDataset, dataset object for testing
    TEXT
        torchtext.data.Field, defines how to convert a text column data into Tensor
    LABEL
        torchtext.data.TabularDataset, defines how to convert label columns data into Tensor
    """
    train_dataset, valid_dataset = data.TabularDataset.splits(
        path=folder,
        train=train_fname,
        validation=val_fname,
        format="csv",
        skip_header=True,
        fields=fields,
    )

    test_dataset = None
    if test_fname:
        test_dataset = data.TabularDataset(
            path=os.path.join(folder, test_fname),
            format="csv",
            skip_header=True,
            fields=fields,
        )

    if vocab.lower() == "glove":
        TEXT.build_vocab(train_dataset, vectors=GloVe(name="6B", dim=300))
    elif vocab.lower() == "fasttext":
        TEXT.build_vocab(train_dataset, vectors=FastText())
    else:
        raise ValueError(
            "Pretrained embeddings of GloVe or FastText are the only options."
        )

    return train_dataset, valid_dataset, test_dataset, TEXT, LABEL


def get_nlp_dataloader(
    train_dataset,
    valid_dataset,
    test_dataset=None,
    batch_size=64,
    text_col="text",
    label_cols=[],
):
    """
    Builds library with a given vocabulary. Forms train, validation and test datasets.

    Parameters
    ----------
    train_dataset
        torchtext.data.TabularDataset, dataset object for training
    valid_dataset
        torchtext.data.TabularDataset, dataset object for validation
    test_dataset
        torchtext.data.TabularDataset, dataset object for testing
    batch_size
        int, batch size
    id_col
        string, name of the id column
    text_col
        string, name of the input text column
    label_cols
        list of strings, names of the label columns in a list
    

    Returns
    -------
    train_dataloader
        BatchWrapper, iterable train dataset
    valid_dataloader
        BatchWrapper, iterable validdation dataset
    test_dataloader
        BatchWrapper, iterable test dataset
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_iter, val_iter = BucketIterator.splits(
        (train_dataset, valid_dataset),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False,
    )
    train_dataloader = BatchWrapper(train_iter, text_col, label_cols)
    valid_dataloader = BatchWrapper(val_iter, text_col, label_cols)

    test_iter = None
    test_dataloader = None
    if test_dataset:
        test_iter = Iterator(
            test_dataset,
            batch_size=batch_size,
            device=device,
            sort=False,
            sort_within_batch=False,
            repeat=False,
        )
        test_dataloader = BatchWrapper(test_iter, text_col, label_cols)

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":

    folder = "~/Oyku/Vela/org_data"
    id_col = "id"
    text_col = "text"
    train_fname = "train.csv"
    val_fname = "val.csv"
    test_fname = "val.csv"
    batch_size = 64

    train_df = pd.read_csv(os.path.join(folder, "train.csv"))
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
