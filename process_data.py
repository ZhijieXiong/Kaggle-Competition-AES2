import argparse
import os

from copy import deepcopy

from util import set_seed, load_csv, load_table


def process_asap_sas(source_dir):
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(raw_data_dir)), "processed")
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    full_data = load_table(os.path.join(source_dir, "train_rel_2.tsv"))
    data_all = deepcopy(full_data)
    data_all["essay_index"] = range(len(data_all))
    data_all["essay_id"] = data_all["Id"]
    data_all["full_text"] = data_all["EssayText"]
    data_all["score"] = data_all["Score1"]
    data_all = data_all[["essay_id", "essay_index", "full_text", "score"]]

    processed_data_dir = os.path.join(processed_dir, f"kaggle-AES2012-all")
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    data_all.to_csv(os.path.join(processed_data_dir, "data.csv"), index=False)

    for set_idx in range(1, 9):
        data = deepcopy(full_data[full_data["EssaySet"] == set_idx])
        data["essay_index"] = range(len(data))
        data["essay_id"] = data["Id"]
        data["full_text"] = data["EssayText"]
        data["score"] = data["Score1"]
        data = data[["essay_id", "essay_index", "full_text", "score"]]

        processed_data_dir = os.path.join(processed_dir, f"kaggle-AES2012-set{set_idx}")
        if not os.path.exists(processed_data_dir):
            os.mkdir(processed_data_dir)
        data.to_csv(os.path.join(processed_data_dir, "data.csv"), index=False)


def process_kaggle_AES2(source_dir):
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(raw_data_dir)), "processed")
    processed_data_dir = os.path.join(processed_dir, "kaggle-AES2024")
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)

    src_data = load_csv(os.path.join(source_dir, "train.csv"))
    src_data["essay_index"] = range(len(src_data))
    src_data.to_csv(os.path.join(processed_data_dir, "data.csv"), index=False)


PROCESS_FUNCTION_TABLE = {
    "asap-sas": process_asap_sas,
    "learning-agency-lab-automated-essay-scoring-2": process_kaggle_AES2
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default=r"F:\code\myProjects\kaggle-AES2-competition\data\raw\asap-sas")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)

    set_seed(params["seed"])

    raw_data_dir = params["raw_data_dir"]
    raw_data_name = os.path.basename(raw_data_dir)
    process_function = PROCESS_FUNCTION_TABLE[raw_data_name]
    process_function(raw_data_dir)
