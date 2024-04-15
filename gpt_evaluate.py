from util import load_json
from train_and_eval.gpt import evaluate


if __name__ == "__main__":
    json_file_path = r"F:\code\myProjects\kaggle-AES2-competition\output\gpt-4-response\zero_shot.json"
    gpt_response = load_json(json_file_path)
    evaluate_result = evaluate(gpt_response)
    print(f"num of sample: {len(gpt_response)}, QWK: {evaluate_result['QWK']}, ACC: {evaluate_result['ACC']}, F1 Score: {evaluate_result['F1']}")
