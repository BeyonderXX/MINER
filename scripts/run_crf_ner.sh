#!/bin/bash

DATA_DIR='/root/RobustNER/data/conll2003/'
MODEL_TYPE='roberta'
MODEL_NAME_OR_PATH='/root/MODELS/roberta-base/'
OUTPUT_DIR='/root/RobustNER/out/roberta/'
LABEL='/root/RobustNER/data/conll2003/labels.txt'


cd /root/RobustNER/scripts

CUDA_VISIBLE_DEVICES='1' python ../examples/run_crf_ner.py \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--labels $LABEL \
--overwrite_output_dir \
--do_train \
--do_eval \
--evaluate_during_training \
--adv_training fgm \
--epoch 3 \
--max_seq_length 128 \
--logging_steps 0.2 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 5e-5 \
--bert_lr 5e-5 \
--classifier_lr  5e-5 \
--crf_lr  1e-3 \
