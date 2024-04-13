#!/usr/bin/env bash

{
  CUDA_VISIBLE_DEVICES=0 python /home/xiongzj/myProjects/kaggle-AES2-competition/bert_classifier_train.py \
    --use_test_dataset False \
    --data_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/data/processed/kaggle-AES2024" \
    --output_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/model_save" \
    --max_label_num 6 \
    --bert_model_dir "/home/xiongzj/myProjects/kaggle-AES2-competition/model_save/bert-base-uncased" \
    --seed 0 \
    --pooling "max" \
    --batch_size 16 \
    --num_epoch 20 \
    --learning_rate 0.00001 \
    --epsilon 0.000001 \
    --warmup_steps 0
} >> /home/xiongzj/myProjects/kaggle-AES2-competition/results/bert-classifier-uncased_kaggle-AES2024_only-valid.txt

