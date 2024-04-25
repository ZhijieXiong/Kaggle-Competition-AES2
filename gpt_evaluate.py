from util import load_json, re_extract_score
from train_and_eval.gpt import evaluate


if __name__ == "__main__":
    json_file_path = r"F:\code\myProjects\kaggle-AES2-competition\output\llama2-7b-hf-response\zero_shot_v2.json"
    generated_output = load_json(json_file_path)
    for k, v in generated_output.items():
        if "score_predict" not in v.keys():
            score_predict = re_extract_score(v["generated_response"])
            assert score_predict is not None, f"The predicted score of essay {k} is not be extracted successfully"
            v["score_predict"] = score_predict
    evaluate_result = evaluate(generated_output)
    print(f"num of sample: {len(generated_output)}, QWK: {evaluate_result['QWK']}, ACC: {evaluate_result['ACC']}, F1 Score: {evaluate_result['F1']}")
