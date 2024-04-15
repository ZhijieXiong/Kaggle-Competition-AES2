import argparse
import os.path

import pandas as pd
import torch

from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

from dataset.BERTDataset import BERTDataset
from model.BERTClassifier import BERTClassifier
from train_and_eval.bert import inference

from util import set_seed, load_csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\data\raw\learning-agency-lab-automated-essay-scoring-2\test.csv")
    parser.add_argument("--model_path", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\model_save\server-BertClassifier@@bert-base-uncased@@seed_42@@kaggle-AES2024@@2024-04-14@01-51-03\epoch-4.pt")
    parser.add_argument("--bert_model_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\model_save\bert-base-uncased")
    parser.add_argument("--max_label_num", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooling", type=str, default="max", choices=("max", "mean"))
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])

    bert_tokenizer = BertTokenizer.from_pretrained(params["bert_model_dir"])
    data = load_csv(params["test_data_path"])
    dim_label = params["max_label_num"]

    X = data["full_text"].values
    y = [0] * len(X)
    test_dataset = BERTDataset(
        txt_list=X.tolist(),
        labels=y,
        tokenizer=bert_tokenizer
    )
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=params["batch_size"])

    model = BERTClassifier(params["bert_model_dir"], freeze_bert=False, dim_out=dim_label, pooling=params["pooling"]).\
        to(DEVICE)
    model.load_state_dict(torch.load(params["model_path"]))
    predict_score = inference(model, test_dataloader, DEVICE)
    submission = pd.DataFrame()
    submission["essay_id"] = data["essay_id"]
    submission["score"] = predict_score
    submission["score"] = submission["score"] + 1
    model_dir = os.path.dirname(params["model_path"])
    submission.to_csv(os.path.join(model_dir, "submission.csv"), index=False)
