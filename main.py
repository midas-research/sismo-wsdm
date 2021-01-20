import argparse
import copy
import json
import os
import pickle
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (AdamW, get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)
import optuna

from dataset import RedditDataset
from loss import loss_function, true_metric_loss
from model import RedditModel
from utils import class_FScore, gr_metrics, make_31, pad_collate_reddit, splits

np.set_printoptions(precision=5)


def train_loop(model, expt_type, dataloader, optimizer, device, dataset_len, loss_type, scale=1):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for bi, inputs in enumerate(dataloader):
        optimizer.zero_grad()

        labels, tweet_features, lens = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)

        output = model(tweet_features, lens, labels)

        _, preds = torch.max(output, 1)

        loss = loss_function(output, labels, loss_type, expt_type, scale)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / dataset_len

    return epoch_loss, epoch_acc


def eval_loop(model, expt_type, dataloader, device, dataset_len, loss_type, scale=1):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    fin_conf = []

    for bi, inputs in enumerate(dataloader):
        labels, tweet_features, lens = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)

        with torch.no_grad():
            output = model(tweet_features, lens, labels)

        _, preds = torch.max(output, 1)

        loss = loss_function(output, labels, loss_type, expt_type, scale)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        fin_conf.append(output.cpu().detach().numpy())

        fin_targets.append(labels.cpu().detach().numpy())
        fin_outputs.append(preds.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets), fin_conf


def main(config):
    expt_type = config['expt_type']

    batch_size = config['batch_size']

    epochs = config['epochs']

    hidden_dim = config['hidden_dim']
    embedding_dim = config['embed_dim']

    num_layers = config['num_layers']
    dropout = config['dropout']
    dist_values = [15, 21, 42, 13, 9]
    learning_rate = config['learning_rate']
    scale = config['scale']

    loss_type = "OE"
    model_type = config['model_type']

    number_of_runs = config['num_runs']

    metrics_dict = {}

    data_dir = config['data_dir']

    for i in trange(number_of_runs):
        data_name = os.path.join(data_dir, f'reddit-longformer.pkl')

        with open(data_name, 'rb') as f:
            df = pickle.load(f)

            if expt_type == 4:
                df['label'] = df['label'].apply(make_31)

        df_train, df_test, _, __ = train_test_split(
            df, df['label'].tolist(), test_size=0.2, stratify=df['label'].tolist())

        train_dataset = RedditDataset(
            df_train.label.values, df_train.enc.values)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=pad_collate_reddit, shuffle=True)

        test_dataset = RedditDataset(df_test.label.values, df_test.enc.values)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=pad_collate_reddit)

        if config['model_type'] == 'lstm+att':
            model = RedditModel(expt_type, embedding_dim,
                                hidden_dim, num_layers, dropout)
        elif config['model_type'] == 'lstm':
            model = RedditNoAttModel(
                expt_type, embedding_dim, hidden_dim, num_layers, dropout)
        elif config['model_type'] == 'avg-pool':
            model = BertPoolRedditModel(expt_type, embedding_dim, dropout)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        optimizer = Adam(model.parameters(),
                         lr=learning_rate)

        scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=epochs)

        best_metric = 0.0
        tc = time.time()

        early_stop_counter = 0
        early_stop_limit = config['early_stop']

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = np.inf

        for epoch in trange(epochs, leave=False):
            loss, accuracy = train_loop(model,
                                        expt_type,
                                        train_dataloader,
                                        optimizer,
                                        device,
                                        len(train_dataset),
                                        loss_type,
                                        scale)


            if scheduler is not None:
                scheduler.step()

            if loss >= best_loss:
                early_stop_counter += 1
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                best_loss = loss

            if early_stop_counter == early_stop_limit:
                break

        tc = time.time()
        model.load_state_dict(best_model_wts)
        _, _, y_pred, y_true, conf = eval_loop(model,
                                               expt_type,
                                               test_dataloader,
                                               device,
                                               len(test_dataset),
                                               loss_type,
                                               scale)

        m = gr_metrics(y_pred, y_true)
        classwise_FScores = class_FScore(y_pred, y_true, expt_type)
        if 'Precision' in metrics_dict:
            metrics_dict['Precision'].append(m[0])
            metrics_dict['Recall'].append(m[1])
            metrics_dict['FScore'].append(m[2])
        else:
            metrics_dict['Precision'] = [m[0]]
            metrics_dict['Recall'] = [m[1]]
            metrics_dict['FScore'] = [m[2]]

    df = pd.DataFrame(metrics_dict)
    df.to_csv(
        f'{datetime.now().__format__("%d%m%y_%H%M%S")}_df.csv')

    return df['FScore'].median()


if __name__ == "__main__":
    model_types = ('avg-pool', 'lstm+att', 'lstm')

    experiment_type = (4, 5)

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--expt-type", type=int, default=5, choices=experiment_type,
                        help="expt type")

    parser.add_argument("--batch-size", type=int, default=8,
                        help="batch size")

    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs")

    parser.add_argument("--num-runs", type=int, default=50,
                        help="number of runs")

    parser.add_argument("--early-stop", type=int, default=10,
                        help="early stop limit")

    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="hidden dimensions")

    parser.add_argument("--embed-dim", type=int, default=768,
                        help="embedding dimensions")

    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of layers")

    parser.add_argument("--dropout", type=float, default=0.3,
                        help="dropout probablity")

    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate")

    parser.add_argument("--scale", type=float, default=1.8,
                        help="scale factor alpha")

    parser.add_argument("--data-dir", type=str, default="",
                        help="directory for data")

    parser.add_argument("--model-type", type=str, default="lstm+att",
                        choices=model_types, help="type of model")

    args = parser.parse_args()
    main(args.__dict__)
