import os
import json
import inspect
import time

from llm_api import prompt_chat
from util import load_csv, load_json, write_json

# 导入key
with open(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), "./prompts.json"), "r") as f:
    PROMPTS = json.load(f)

MODEL_NAME = "gpt-4"
DATA = load_csv("./data/processed/kaggle-AES2024/data.csv")
# 在数据的预处理中都减了1
DATA["score"] = DATA["score"] + 1
DATA_LIST = DATA.to_dict(orient="records")
DATA_DICT = {row_data["essay_id"]: row_data for row_data in DATA_LIST}
OUTPUT_DIR = f"./output/{MODEL_NAME}-response"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


if __name__ == "__main__":
    target_essay_ids = ["0033037", "001ab80", "06e2db4"]
    num_call_api = 3

    # 给LLM文章和分数，让LLM推断打这个分数的理由
    prompt_name = "zero_shot_v2"
    cot_from_llm_zero_shot_path = os.path.join(OUTPUT_DIR, f"cot_from_llm-{prompt_name}.json")
    cot_from_llm_zero_shot = {}
    if os.path.exists(cot_from_llm_zero_shot_path):
        cot_from_llm_zero_shot = load_json(cot_from_llm_zero_shot_path)
    num_called_api = 0
    for essay_id in target_essay_ids:
        if num_called_api >= num_call_api:
            break

        if essay_id not in DATA_DICT.keys():
            continue

        if essay_id in cot_from_llm_zero_shot.keys():
            continue

        essay_data = DATA_DICT[essay_id]
        full_text = essay_data["full_text"]
        ground_truth_score = essay_data["score"]
        prompt = f"{PROMPTS['cot_from_gpt'][prompt_name]}\n\n" \
                 f"The following is the content of the essay:\n\n" \
                 f"{full_text}\n\n" \
                 f"And this is the score given by expert: {ground_truth_score}\n\n" \
                 f"Now please tell me the reason why the senior marker gave this score.\n\n"
        gpt_response = prompt_chat(MODEL_NAME, prompt).content
        cot_from_llm_zero_shot[essay_id] = {
            "full_text": full_text,
            "generated_cot": gpt_response,
            "score_ground_truth": essay_data["score"],
        }
        num_called_api += 1
        time.sleep(1)
    write_json(cot_from_llm_zero_shot, cot_from_llm_zero_shot_path)
