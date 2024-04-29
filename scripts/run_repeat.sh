#!/usr/bin/env bash

for ((i=1; i<=20; i++))
do
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v1_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_1-shot-low-score_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_1-shot-middle-score_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_1-shot-high-score_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_2-shot-low-high-score_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_2-shot-low-middle-score_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_2-shot-middle-high-score_kaggle_AES2.sh
  bash /home/xiongzj/myProjects/kaggle-AES2-competition/scripts/llama3-8b-instruct_zero-shot-v2_3-shot_kaggle_AES2.sh
done
