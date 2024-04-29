import os
import json
import inspect
import time

from llm_api import prompt_chat
from util import load_csv, load_json, write_json, extract_score

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


if __name__ == "__main__":
    num_call_api = 1

    # 使用GPT提炼Holistic Rating for Source-Based Writing和Holistic Rating for Independent Writing
    # summary4source_based_writing = prompt_chat(MODEL_NAME, PROMPTS["summary"]["rating4source_based_writing"])
    # summary4independent_writing = prompt_chat(MODEL_NAME, PROMPTS["summary"]["rating4independent_writing"])

    # zero shot
    # prompt_name = "zero_shot_v2"
    # zero_shot_response_path = os.path.join(OUTPUT_DIR, f"{prompt_name}.json")
    # zero_shot_response = {}
    # if os.path.exists(zero_shot_response_path):
    #     zero_shot_response = load_json(zero_shot_response_path)
    # num_called_api = 0
    # for i, (_, row_data) in enumerate(DATA.iterrows()):
    #     if num_called_api >= num_call_api:
    #         break
    #
    #     essay_id = row_data["essay_id"]
    #     if essay_id in zero_shot_response.keys():
    #         continue
    #
    #     full_text = row_data["full_text"]
    # prompt = f"{PROMPTS[prompt_name]}\n\n" \
    #          f"The following is the content of the essay:\n\n" \
    #          f"{full_text}\n\n" \
    #          f"Please give your score firstly (the score is wrapped in `$$`). " \
    #          f"Then give the reason for this score. " \
    #          f"For example: `I would give this essay a score of $$4$$. The reason for giving this score is ...`\n\n"
    #     has_error, gpt_response, score = extract_score(prompt_chat(MODEL_NAME, prompt).content)
    #     zero_shot_response[essay_id] = {
    #         "full_text": full_text,
    #         "generated_response": gpt_response,
    #         "score_ground_truth": row_data["score"],
    #     }
    #     if not has_error:
    #         zero_shot_response[essay_id]["score_predict"] = score
    #     num_called_api += 1
    #     time.sleep(1)
    # write_json(zero_shot_response, zero_shot_response_path)

    # few shot
    prompt_name = "zero_shot_v2"
    example_name = "1_shot_low_score"
    few_shot_response_path = os.path.join(OUTPUT_DIR, f"few_shot-using_{prompt_name}-{example_name}.json")
    few_shot_response = {}
    if os.path.exists(few_shot_response_path):
        few_shot_response = load_json(few_shot_response_path)
    num_called_api = 0
    example_essay_id = PROMPTS["few_shot_example_essay_id"][example_name]
    for i, (_, row_data) in enumerate(DATA.iterrows()):
        if num_called_api >= num_call_api:
            break

        essay_id = row_data["essay_id"]
        if (essay_id in few_shot_response.keys()) or (essay_id == example_essay_id):
            continue

        full_text = row_data["full_text"]
        if "1_shot" in example_name:
            prompt = f"{PROMPTS[prompt_name]}\n\n" \
                     f"This is an example:\n\n" \
                     f"{PROMPTS[example_name]}\n\n" \
                     f"The following is the content of the essay:\n\n" \
                     f"{full_text}\n\n" \
                     f"Please give your score firstly (the score is wrapped in `$$`). " \
                     f"Then give the reason for this score. " \
                     f"For example: `I would give this essay a score of $$4$$. The reason for giving this score is ...`\n\n"
        else:
            prompt = f"{PROMPTS[prompt_name]}\n\n" \
                     f"There are some examples:\n\n" \
                     f"{PROMPTS[example_name]}\n\n" \
                     f"The following is the content of the essay:\n\n" \
                     f"{full_text}\n\n" \
                     f"Please give your score firstly (the score is wrapped in `$$`). " \
                     f"Then give the reason for this score. " \
                     f"For example: `I would give this essay a score of $$4$$. The reason for giving this score is ...`\n\n"

        has_error, gpt_response, score = extract_score(prompt_chat(MODEL_NAME, prompt).content)
        few_shot_response[essay_id] = {
            "full_text": full_text,
            "generated_response": gpt_response,
            "score_ground_truth": row_data["score"],
        }
        if not has_error:
            few_shot_response[essay_id]["score_predict"] = score
        num_called_api += 1
        time.sleep(1)
    write_json(few_shot_response, few_shot_response_path)
