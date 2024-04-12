import argparse
import os.path

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import random
import numpy as np

from dataset.BERTDataset import BERTDataset
from model.BERTClassifier import BERTClassifier
from train_and_eval.bert import train

from util import get_now_time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_number", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)

    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=0)
    args = parser.parse_args()
    params = vars(args)

    set_seed(params["seed"])

    data_path = r"F:\code\myProjects\kaggle-AES2-competition\data\raw\asap-sas\train_rel_2.tsv"
    bert_model_dir = r"F:\code\myProjects\kaggle-AES2-competition\model_save\bert-base-uncased"

    full_data = pd.read_table(data_path)
    data = full_data[full_data["EssaySet"] == params["set_number"]]
    data.index = range(len(data))
    dim_label = len(data["Score1"].drop_duplicates())

    X = data["EssayText"].values
    y = data["Score1"].values
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.2, random_state=params["seed"])
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=params["seed"])

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    train_dataset = BERTDataset(
        txt_list=X_train.tolist(),
        labels=y_train.tolist(),
        tokenizer=bert_tokenizer,
        max_length=params["max_len"]
    )
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=params["batch_size"], shuffle=True)
    valid_dataset = BERTDataset(
        txt_list=X_val.tolist(),
        labels=y_val.tolist(),
        tokenizer=bert_tokenizer,
        max_length=params["max_len"]
    )
    valid_dataloader = DataLoader(valid_dataset, sampler=RandomSampler(valid_dataset), batch_size=params["batch_size"])
    test_dataset = BERTDataset(
        txt_list=X_test.tolist(),
        labels=y_test.tolist(),
        tokenizer=bert_tokenizer,
        max_length=params["max_len"]
    )
    test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=params["batch_size"])

    model = BERTClassifier(bert_model_dir, freeze_bert=False, dim_out=dim_label).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"], eps=params["epsilon"])
    total_steps = len(train_dataloader) * params["num_epoch"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["warmup_steps"],
        num_training_steps=total_steps
    )

    output_dir_name = f"BertClassifier@@seed_{params['seed']}@@asap-sas-set{params['set_number']}@@" \
                 f"{get_now_time().replace(' ', '@').replace(':', '-')}"
    output_dir = os.path.join(r"F:\code\myProjects\kaggle-AES2-competition\model_save", output_dir_name)
    os.mkdir(output_dir)
    train(model, train_dataloader, valid_dataloader, test_dataloader, optimizer, scheduler, params["num_epoch"], DEVICE,
          output_dir, evaluation=True)
