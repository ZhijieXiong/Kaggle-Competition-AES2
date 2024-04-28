#!/usr/bin/env bash

{
  CUDA_VISIBLE_DEVICES=1 python /home/xiongzj/myProjects/kaggle-AES2-competition/bert_classifier_train.py \
    --use_test_dataset False \
    --data_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/data/processed/kaggle-AES2024" \
    --output_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/model_save" \
    --max_label_num 6 \
    --bert_model_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/model_save/bert-base-cased" \
    --seed 42 \
    --pooling "max" \
    --batch_size 16 \
    --num_epoch 10 \
    --learning_rate 0.00001 \
    --epsilon 0.000001 \
    --warmup_steps 0
} >> /home/xiongzj/myProjects/kaggle-AES2-competition/results/bert-classifier-cased_kaggle-AES2024_only-valid_seed-42.txt

