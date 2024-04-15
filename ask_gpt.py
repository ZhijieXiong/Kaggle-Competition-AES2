import os
import json
import inspect
import re
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
OUTPUT_DIR = f"./output/{MODEL_NAME}-response"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def extract_score_explanation(gpt_response_):
    try:
        score_ = int(re.search("\$\$(\d)\$\$", gpt_response_).groups()[0])
        explanation_ = gpt_response_.replace(f"$${score_}$$", "").strip(" ").strip(".").strip(",").strip(" ").strip(".").strip(",")
        return False, gpt_response_, score_, explanation_
    except:
        return True, gpt_response_, None, None


if __name__ == "__main__":
    num_call_api = 50

    # 使用GPT提炼Holistic Rating for Source-Based Writing和Holistic Rating for Independent Writing
    # summary4source_based_writing = prompt_chat(MODEL_NAME, PROMPTS["summary"]["rating4source_based_writing"])
    # summary4independent_writing = prompt_chat(MODEL_NAME, PROMPTS["summary"]["rating4independent_writing"])

    # zero shot
    zero_shot_response_path = os.path.join(OUTPUT_DIR, "zero_shot.json")
    zero_shot_response = {}
    if os.path.exists(zero_shot_response_path):
        zero_shot_response = load_json(zero_shot_response_path)
    num_called_api = 0
    for i, (_, row_data) in enumerate(DATA.iterrows()):
        if num_called_api >= num_call_api:
            break

        essay_id = row_data["essay_id"]
        if essay_id in zero_shot_response.keys():
            continue

        full_text = row_data["full_text"]
        prompt = PROMPTS["zero_shot4AES"] + "The following is the content of the essay:\n" + full_text + "\n"
        has_error, gpt_response, score, explanation = extract_score_explanation(prompt_chat(MODEL_NAME, prompt).content)
        zero_shot_response[essay_id] = {
            "full_text": full_text,
            "gpt_response": gpt_response,
            "score_ground_truth": row_data["score"],
        }
        if not has_error:
            zero_shot_response[essay_id]["score_predict"] = score
            zero_shot_response[essay_id]["chain_of_thought"] = explanation
        num_called_api += 1
        time.sleep(3)
    write_json(zero_shot_response, zero_shot_response_path)
