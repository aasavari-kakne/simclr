#!/bin/sh

# Run this from within docker

python run_pretrain.py --train_mode=pretrain \
  --train_batch_size=4 --train_epochs=100 \
  --learning_rate=1.0 --weight_decay=1e-6 --temperature=0.5 \
  --image_size=224 --eval_split=test --resnet_depth=50 --width_multiplier=4 \
  --use_blur=False --color_jitter_strength=0.5 \
  --data_dir=/Users/amagnan/Documents/walmart/product_data/simclr/data \
  --model_dir=/Users/amagnan/Documents/walmart/product_data/simclr/pretrain_model --use_tpu=False |& tee pretrain.log
