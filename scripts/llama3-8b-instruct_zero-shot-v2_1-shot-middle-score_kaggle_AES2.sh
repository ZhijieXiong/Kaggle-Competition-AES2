#!/usr/bin/env bash

{
  CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1  /home/xiongzj/myProjects/kaggle-AES2-competition/ask_local_llm.py \
    --prompt_type "few_shot" --zero_shot_type "zero_shot_v2" --example_type "1_shot_middle_score" \
    --prompts_path "/home/xiongzj/myProjects/kaggle-AES2-competition/prompts.json" \
    --model_dir "/data/xiongzj/LLM/llama3-8b-instruct" \
    --data_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/data/processed" \
    --output_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/output" \
    --data_name "kaggle-AES2024" --num_ask 50 \
    --temperature 0.6 --top_p 0.9 --max_seq_len 8192 --max_batch_size 8
} >> /home/xiongzj/myProjects/kaggle-AES2-competition/results/llama3-8b-instruct_zero-shot-v2_1-shot-middle-score_kaggle-AES2024.txt

