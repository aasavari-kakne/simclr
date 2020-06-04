#!/bin/sh

# Run this from within docker

python run_finetune.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.05 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=256 --warmup_epochs=0 \
  --image_size=224 --eval_split=test --resnet_depth=50 --width_multiplier=4 \
  --data_dir=/data/finetune_dataset \
  --checkpoint=/data/checkpoint/ResNet50_4x --model_dir=/data/finetune_model --use_tpu=False
