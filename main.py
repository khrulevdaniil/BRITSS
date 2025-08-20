import copy
import time
import argparse

import numpy as np
import pandas as pd
import ujson as json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from ipdb import set_trace

import utils
import models
import data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_iter = data_loader.get_loader(batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)

            # .item() вместо устаревшего .data[0]
            run_loss += ret['loss'].item()

            pct = (idx + 1) * 100.0 / len(data_iter)
            avg_loss = run_loss / (idx + 1.0)
            print(f"\rProgress epoch {epoch}, {pct:.2f}%, average loss {avg_loss:.6f}",
                  end='', flush=True)

        print()  # перенос строки после прогресс-бара

        # В оригинале evaluate вызывался каждый эпох
        if epoch % 1 == 0:
            evaluate(model, data_iter)


def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []
    evals = []
    imputations = []

    with torch.no_grad():
        for idx, data in enumerate(val_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, None)

            pred = ret['predictions'].data.cpu().numpy()
            label = ret['labels'].data.cpu().numpy()
            is_train = ret['is_train'].data.cpu().numpy()

            eval_masks = ret['eval_masks'].data.cpu().numpy()
            eval_ = ret['evals'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()

            evals += eval_[np.where(eval_masks == 1)].tolist()
            imputations += imputation[np.where(eval_masks == 1)].tolist()

            # collect test label & prediction
            pred = pred[np.where(is_train == 0)]
            label = label[np.where(is_train == 0)]

            labels += label.tolist()
            preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    auc = metrics.roc_auc_score(labels, preds)
    print(f"AUC {auc:.6f}")

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    mae = np.abs(evals - imputations).mean()
    mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()

    print(f"MAE {mae:.6f}")
    print(f"MRE {mre:.6f}")


def run():
    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)


if __name__ == '__main__':
    run()

