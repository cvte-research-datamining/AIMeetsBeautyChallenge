import faiss
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import pickle
import csv
import pandas as pd
import random
from pathlib import Path
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Perform the evaluation on prediction results.")
    parser.add_argument("--prediction-path", type=str, default='./results/predictions.csv', help="(default:./results/predictions.csv")
    parser.add_argument("--label-path", type=str, default='./eval_label.csv', help="(default:./eval_label.csv)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    labels = pd.read_csv(args.label_path, encoding='utf-8')
    predictions = pd.read_csv(args.prediction_path, encoding='utf-8', header=None)
    labels.columns = ['Query'] + list(range(len(labels.columns)-1))
    predictions.columns = ['Query'] + list(range(len(predictions.columns)-1))
    labels = labels.set_index('Query')
    predictions = predictions.set_index('Query')

    # Compute the mean average precision
    query_ids = predictions.index.values.tolist()
    AP = []
    for id_ in query_ids:
        pred = predictions.loc[id_].dropna().values.tolist()
        label = labels.loc[id_].dropna().values.tolist()
        # label.append(id_)

        precision = []
        num_correct = 0
        position = 0
        for p in pred:
            position += 1
            if p in label:
                num_correct += 1
                precision.append(num_correct / position)
            else:
                precision.append(0)
        if num_correct != 0:
            AP.append(np.array(precision).sum() / num_correct)
        else:
            AP.append(0)
        print(id_, precision, AP[-1])
    mAP = np.array(AP).mean()
    print('Mean Average Precision:{:}'.format(mAP))

if __name__ == '__main__':
    main()