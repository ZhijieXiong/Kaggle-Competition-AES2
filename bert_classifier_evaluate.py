import argparse
import os
import torch

from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from dataset.BERTDataset import BERTDataset
from model.BERTClassifier import BERTClassifier
from train_and_eval.bert import evaluate

from util import get_now_time, set_seed, load_csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\data\processed\kaggle-AES2024")
    parser.add_argument("--model_path", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\model_save\BertClassifier@@seed_0@@kaggle-AES2024@@2024-04-13@14-43-14\epoch-2.pt")
    parser.add_argument("--bert_model_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\model_save\bert-base-uncased")
    parser.add_argument("--output_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\model_save")
    parser.add_argument("--max_label_num", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooling", type=str, default="max", choices=("max", "mean"))
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])

    bert_tokenizer = BertTokenizer.from_pretrained(params["bert_model_dir"])
    data_dir = params["data_dir"]
    data_name = os.path.basename(data_dir)
    data = load_csv(os.path.join(data_dir, "data.csv"), num_rows=500)
    dim_label = params["max_label_num"]

    X = data["full_text"].values
    y = data["score"].values
    _, X_other, _, y_other = train_test_split(X, y, test_size=0.3, random_state=params["seed"])
    _, X_test, _, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=params["seed"])
    test_dataset = BERTDataset(
        txt_list=X_test.tolist(),
        labels=y_test.tolist(),
        tokenizer=bert_tokenizer
    )
    test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=params["batch_size"])

    model = BERTClassifier(params["bert_model_dir"], freeze_bert=False, dim_out=dim_label, pooling=params["pooling"]).\
        to(DEVICE)
    model.load_state_dict(torch.load(params["model_path"]))
    evaluate(model, test_dataloader, DEVICE)
