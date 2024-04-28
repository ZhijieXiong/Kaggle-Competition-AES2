import json
import argparse
import os
import transformers
import torch

from transformers import AutoTokenizer

from model.llama2.generation import Llama as Llama2
from model.llama3.generation import Llama as Llama3
from util import load_csv, load_json, write_json, get_now_time, extract_score


MODEL_NAME2TYPE = {
    "llama2": Llama2,
    "llama3": Llama3
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\prompts.json")
    parser.add_argument("--model_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\llama2-7b-hf")
    parser.add_argument("--data_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\data\processed")
    parser.add_argument("--output_dir", type=str,
                        default=r"F:\code\myProjects\kaggle-AES2-competition\output")
    parser.add_argument("--data_name", type=str, default="kaggle-AES2024")
    parser.add_argument("--num_ask", type=int, default=10)

    # model params
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_batch_size", type=int, default=4)

    args = parser.parse_args()
    params = vars(args)

    # ---------------------------------------参数处理---------------------------------------
    with open(params["prompts_path"], "r") as f:
        PROMPTS = json.load(f)

    DATA = load_csv(os.path.join(params["data_dir"], params["data_name"], "data.csv"))
    DATA["score"] = DATA["score"] + 1

    MODEL_DIR = params["model_dir"]

    # 判断模型是否可用
    MODEL_NAME = os.path.basename(MODEL_DIR)
    assert os.path.exists(MODEL_DIR), f"{MODEL_NAME} is not exist"
    llm_name = MODEL_NAME.split("-")[0]
    assert llm_name in MODEL_NAME2TYPE.keys(), f"{llm_name} is not supported"

    IS_HF = MODEL_NAME.split("-")[-1] == "hf"
    OUTPUT_DIR = os.path.join(params["output_dir"], f"{MODEL_NAME}-response")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    NUM_ASK = params["num_ask"]
    # -------------------------------------------------------------------------------------

    # load pipline or generator
    if IS_HF:
        model = transformers.pipeline(
            "text-generation",
            model=MODEL_DIR,
            torch_dtype=torch.float16,
            # Do not use `device_map` AND `device` at the same time as they will conflict
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        model = MODEL_NAME2TYPE[llm_name].build(
            ckpt_dir=MODEL_DIR,
            tokenizer_path=os.path.join(MODEL_DIR, "tokenizer.model"),
            max_seq_len=params["max_seq_len"],
            max_batch_size=params["max_batch_size"],
        )
        tokenizer = None

    # zero shot
    prompt_name = "zero_shot_v2"
    zero_shot_response_path = os.path.join(OUTPUT_DIR, f"{prompt_name}.json")
    zero_shot_response = {}
    if os.path.exists(zero_shot_response_path):
        zero_shot_response = load_json(zero_shot_response_path)

    num_asked = 0
    dialogs = []
    essay_ids2query = []
    for i, (_, row_data) in enumerate(DATA.iterrows()):
        if num_asked >= NUM_ASK:
            break

        essay_id = row_data["essay_id"]
        if essay_id in zero_shot_response.keys():
            continue

        essay_ids2query.append(essay_id)
        full_text = row_data["full_text"]
        prompt = f"{PROMPTS[prompt_name]}\n\n" \
                 f"The following is the content of the essay:\n\n" \
                 f"{full_text}\n\n" \
                 f"Please give your score firstly (the score is wrapped in `$$`). " \
                 f"Then give the reason for this score. " \
                 f"For example: `I would give this essay a score of $$score_number$$. The reason for giving this score is ...`\n\n"

        if IS_HF:
            sequences = model(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                truncation=False,
                max_length=params["max_seq_len"],
            )
            generated_response = sequences[0]["generated_text"]

            print(f"{get_now_time()}: complete query {i}")
            has_error, _, score = extract_score(generated_response)
            zero_shot_response[essay_id] = {
                "full_text": full_text,
                "generated_response": generated_response,
                "score_ground_truth": row_data["score"],
            }
            if not has_error:
                zero_shot_response[essay_id]["score_predict"] = score
        else:
            dialogs.append([{"role": "user", "content": prompt}])

        num_asked += 1

    if not IS_HF:
        print(f"{get_now_time()}: start inference")
        DATA_LIST = DATA.to_dict(orient="records")
        DATA_DICT = {row_data["essay_id"]: row_data for row_data in DATA_LIST}

        batch_data = []
        essay_ids = []
        max_batch_size = params["max_batch_size"]
        while len(dialogs) > 0:
            batch_data.append(dialogs.pop(0))
            essay_ids.append(essay_ids2query.pop(0))

            if len(batch_data) == max_batch_size or len(dialogs) == 0:
                generated_results = model.chat_completion(
                    batch_data,
                    max_gen_len=None,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                )

                for essay_id, generated_result in zip(essay_ids, generated_results):
                    generated_response = generated_result['generation']['content']
                    has_error, _, score = extract_score(generated_response)
                    zero_shot_response[essay_id] = {
                        "full_text": DATA_DICT[essay_id]["full_text"],
                        "generated_response": generated_response,
                        "score_ground_truth": DATA_DICT[essay_id]["score"],
                    }
                    if not has_error:
                        zero_shot_response[essay_id]["score_predict"] = score

                print(f"{get_now_time()}: complete one batch")
                batch_data = []
                essay_ids = []

    write_json(zero_shot_response, zero_shot_response_path)
