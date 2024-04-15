import argparse
import os
import torch

from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

from dataset.BERTDataset import BERTDataset
from model.BERTClassifier import BERTClassifier
from train_and_eval.bert import train

from util import get_now_time, set_seed, load_csv, str2bool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\data\processed\kaggle-AES2024")
    parser.add_argument("--output_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\model_save")
    parser.add_argument("--max_label_num", type=int, default=6)
    parser.add_argument("--bert_model_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\model_save\bert-base-cased")
    parser.add_argument("--use_test_dataset", type=str2bool, default=False)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pooling", type=str, default="max", choices=("max", "mean"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])

    bert_tokenizer = BertTokenizer.from_pretrained(params["bert_model_dir"])
    dim_label = params["max_label_num"]

    data_dir = params["data_dir"]
    data_train_with_test_path = os.path.join(data_dir, "data_train_with_test.csv")
    data_train_without_test_path = os.path.join(data_dir, "data_train_without_test.csv")
    data_valid_with_test_path = os.path.join(data_dir, "data_valid_with_test.csv")
    data_valid_without_test_path = os.path.join(data_dir, "data_valid_without_test.csv")
    data_test_path = os.path.join(data_dir, "data_test.csv")
    if os.path.exists(data_train_with_test_path) and params["use_test_dataset"] and \
            os.path.exists(data_valid_with_test_path) and os.path.exists(data_test_path):
        data_train = load_csv(data_train_with_test_path)
        data_valid = load_csv(data_valid_with_test_path)
        data_test = load_csv(data_test_path)
        test_dataset = BERTDataset(
            txt_list=data_test["full_text"].values.tolist(),
            labels=data_test["score"].values.tolist(),
            tokenizer=bert_tokenizer
        )
        test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=params["batch_size"])
    elif os.path.exists(data_train_without_test_path) and not params["use_test_dataset"] and \
            os.path.exists(data_valid_without_test_path) and os.path.exists(data_test_path):
        data_train = load_csv(data_train_without_test_path)
        data_valid = load_csv(data_valid_without_test_path)
        test_dataloader = None
    else:
        data = load_csv(os.path.join(data_dir, "data.csv"))
        data_train, data_other = train_test_split(data, test_size=0.3, random_state=0)
        if params["use_test_dataset"]:
            data_valid, data_test = train_test_split(data_other, test_size=0.5, random_state=0)
            test_dataset = BERTDataset(
                txt_list=data_test["full_text"].values.tolist(),
                labels=data_test["score"].values.tolist(),
                tokenizer=bert_tokenizer
            )
            test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=params["batch_size"])
            data_train.to_csv(os.path.join(data_dir, "data_train_with_test.csv"), index=False)
            data_valid.to_csv(os.path.join(data_dir, "data_valid_with_test.csv"), index=False)
            data_test.to_csv(os.path.join(data_dir, "data_test.csv"), index=False)
        else:
            data_valid = data_other
            data_train.to_csv(os.path.join(data_dir, "data_train_without_test.csv"), index=False)
            data_valid.to_csv(os.path.join(data_dir, "data_valid_without_test.csv"), index=False)
            test_dataloader = None
    train_dataset = BERTDataset(
        txt_list=data_train["full_text"].values.tolist(),
        labels=data_train["score"].values.tolist(),
        tokenizer=bert_tokenizer
    )
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=params["batch_size"])
    valid_dataset = BERTDataset(
        txt_list=data_valid["full_text"].values.tolist(),
        labels=data_valid["score"].values.tolist(),
        tokenizer=bert_tokenizer
    )
    valid_dataloader = DataLoader(valid_dataset, sampler=RandomSampler(valid_dataset), batch_size=params["batch_size"])

    model = BERTClassifier(params["bert_model_dir"], freeze_bert=False, dim_out=dim_label, pooling=params["pooling"]).\
        to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"], eps=params["epsilon"])
    total_steps = len(train_dataloader) * params["num_epoch"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["warmup_steps"],
        num_training_steps=total_steps
    )

    bert_name = os.path.basename(params["bert_model_dir"])
    data_name = os.path.basename(data_dir)
    output_dir_name = f"BertClassifier@@{bert_name}@@seed_{params['seed']}@@{data_name}@@" \
                      f"{get_now_time().replace(' ', '@').replace(':', '-')}"
    output_dir = os.path.join(params["output_dir"], output_dir_name)
    os.mkdir(output_dir)
    train(model, train_dataloader, valid_dataloader, test_dataloader, optimizer, scheduler, params["num_epoch"], DEVICE,
          output_dir, evaluation=params["use_test_dataset"])
