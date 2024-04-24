#!/usr/bin/env bash

{
  CUDA_VISIBLE_DEVICES=0 python /home/xiongzj/myProjects/kaggle-AES2-competition/ask_local_llm.py \
    --prompts_path "/home/xiongzj/myProjects/kaggle-AES2-competition/prompts.json" \
    --model_dir "/data/xiongzj/LLM/llama2-7b-chat-hf" \
    --data_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/data" \
    --output_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/output" \
    --data_name "kaggle-AES2024" --num_ask 10 \
    --temperature 0.6 --top_p 0.9 --max_seq_len 4096 --max_batch_size 8
} >> /home/xiongzj/myProjects/kaggle-AES2-competition/results/llama2-7b-chat-hf_kaggle-AES2024.txt

