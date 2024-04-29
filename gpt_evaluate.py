from util import load_json, re_extract_score
from train_and_eval.gpt import evaluate


if __name__ == "__main__":
    json_file_path = r"F:\code\myProjects\kaggle-AES2-competition\output\llama3-8b-instruct-response\zero_shot_v2.json"
    generated_output = load_json(json_file_path)

    num_extracted_fail = 0
    essay_ids_extracted_fail = []
    for k, v in generated_output.items():
        if "score_predict" not in v.keys():
            has_error, generated_response, score_predict = re_extract_score(v["generated_response"])
            if has_error:
                num_extracted_fail += 1
                essay_ids_extracted_fail.append(k)
            else:
                v["score_predict"] = score_predict
    print(f"fail to extract score (num is {num_extracted_fail}): {essay_ids_extracted_fail}")
    for essay_id in essay_ids_extracted_fail:
        del generated_output[essay_id]

    evaluate_result = evaluate(generated_output)
    print(f"num of sample: {len(generated_output)}, QWK: {evaluate_result['QWK']}, ACC: {evaluate_result['ACC']}, F1 Score: {evaluate_result['F1']}")
