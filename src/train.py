''' Trains the CategoryPredictor and writes the best model checkpoints and
the final trained model into an output directory.
'''
import argparse
import os
import json
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

#from tqdm.notebook import tqdm

from data.dataloader import preprocess, get_nlp_dataset, get_nlp_dataloader
from model.models import CategoryPredictor
from util import clip_gradient, get_test_predictions, load_fasttext, label_word_embeddings

def train_epoch(model, train_dataloader, valid_dataloader, label_embed, epochs = 3, learning_rate = 1e-3, clip=0.95, threshold = 0.5, output_dir='./'):
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
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

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
                    print('{} - loss: {}'.format(count, loss.data))
                count += 1
            epoch_loss = running_loss / len(train_dataloader)

            val_loss = 0.0
            model.eval()
            for x, y in valid_dataloader:
                preds = model(x, label_embed)
                loss = criterion(y, preds)
                val_loss += loss.data * x.size(0)

            val_loss /= len(valid_dataloader)

            valid_preds, avg_jaccard = get_test_predictions(model, valid_dataloader, label_embed, threshold = 0.5)
            if avg_jaccard > best_jaccard:
                best_jaccard = avg_jaccard
                torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}.pkl'))
                best_model = model

            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Jaccard Score: {:.4f}'.format(epoch, epoch_loss, val_loss, avg_jaccard))
        torch.save(model.state_dict(), os.path.join(output_dir, 'model_epoch_final.pkl'))

    except KeyboardInterrupt:
        
        print('Saving model before interupting')
        torch.save(model.state_dict(), os.path.join(output_dir,  f'model_epoch_{epoch}.interrupt.pkl'))
    except KeyboardInterrupt:
        
        print('Saving model before interupting')
        torch.save(model.state_dict(), os.path.join(output_dir,  f'model_epoch_{epoch}.interrupt.pkl'))

    return model, best_model


def main(config_path):
    config = json.load(open(config_path, "r"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load parameters from config
    folder = config["paths"]["folder"]
    output_dir = os.path.join(folder, config["paths"]["output_dir"])
    train_fname = config["paths"]["train_filename"]
    val_fname = config["paths"]["val_filename"]
    test_fname = config["paths"]["test_filename"]
    bin_path = config["paths"]["embedding_bin_directory"]

    batch_size = config["train_params"]["batch_size"]
    id_col = config["train_params"]["id_column"]
    text_col = config["train_params"]["text_column"]
    vocab = config["train_params"]["vocab"]
    hidden_dim = config["train_params"]["hidden_dim"]
    embed_dim = config["train_params"]["hidden_dim"]
    epochs = config["train_params"]["epochs"]
    learning_rate = config["train_params"]["learning_rate"]
    clip = config["train_params"]["clip"]
    threshold = config["train_params"]["logit_threshold"]
    pretrained = True if (vocab.lower() == 'fasttext') or (vocab.lower() == 'glove') else False


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Output directory created in: {}'.format(output_dir))

    # preprocessing of data
    train_df = pd.read_csv(os.path.join(folder, train_fname))
    label_cols = [col for col in train_df.columns if not(col == 'Unnamed: 0') and not(col == id_col) and not (col == text_col)]

    fields, TEXT, LABEL = preprocess(train_df, id_col=id_col, text_col=text_col, label_cols=label_cols)
    train_dataset, valid_dataset, test_dataset, TEXT, LABEL = get_nlp_dataset(fields, TEXT, LABEL, vocab=vocab, folder=folder, train_fname=train_fname, val_fname=val_fname, test_fname=test_fname)
    train_dataloader, valid_dataloader, test_dataloader = get_nlp_dataloader(train_dataset, valid_dataset, test_dataset, batch_size, id_col, text_col, label_cols)

    # loads words embeddings
    fasttext_embed = load_fasttext(bin_path)
    label_embed = label_word_embeddings(label_cols, fasttext_embed)

    # fixed model parameters
    num_linear = 1
    vocab_size = len(TEXT.vocab)
    num_class = len(label_cols)
    word_embed = TEXT.vocab.vectors

    # model definition and training
    model = CategoryPredictor(hidden_dim, vocab_size, word_embed, embed_dim, num_linear, num_class, pretrained, lstm_layers=2)
    final_model, best_model = train_epoch(model, train_dataloader, valid_dataloader, label_embed, epochs, learning_rate, clip, threshold, output_dir)
    test_preds, jaccard_index = get_test_predictions(best_model, test_dataloader, label_embed, threshold)

    return final_model, best_model, jaccard_index

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "-c", "--config", help="filepath to config json", default="./config.json"
    )
    ARGS = PARSER.parse_args()
    CONFIGPATH = ARGS.config

    final_model, best_model, jaccard_index = main(CONFIGPATH)
    
