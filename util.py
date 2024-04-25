import datetime
import argparse
import torch
import random
import json
import re
import numpy as np
import pandas as pd


def get_now_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def load_csv(data_path, useful_cols=None, rename_dict=None, num_rows=None):
    try:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False, index_col=False, nrows=num_rows)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, index_col=False, nrows=num_rows)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def load_table(data_path, useful_cols=None, rename_dict=None, num_rows=None):
    try:
        df = pd.read_table(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False, nrows=num_rows)
    except UnicodeDecodeError:
        df = pd.read_table(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, nrows=num_rows)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_json(json_path):
    with open(json_path, "r") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)


def extract_score(generated_response):
    try:
        score = int(re.search("\$\$(\d)\$\$", generated_response).groups()[0])
        return False, generated_response, score
    except:
        return True, generated_response, None


def re_extract_score(generated_response):
    try:
        score1 = re.search("\$\$(\d)\$\$", generated_response).groups()
        score2 = re.search("\$(\d)\$\$", generated_response).groups()
        score3 = re.search("\$\$(\d)\$", generated_response).groups()
        score4 = re.search("\$(\d)\$", generated_response).groups()
        if len(score1) > 0:
            score = score1[0]
        elif len(score2) > 0:
            score = score2[0]
        elif len(score3) > 0:
            score = score3[0]
        elif len(score4) > 0:
            score = score4[0]
        else:
            score = None
        if score is not None:
            return False, generated_response, int(score)
        else:
            return False, generated_response, None
    except:
        return True, generated_response, None
