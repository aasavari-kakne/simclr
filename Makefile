# Use:
#
# `make container` will build a container
#

IMAGE_PARENT ?=hub.docker.prod.walmart.com/tensorflow/tensorflow:1.15.0-gpu-py3
IMAGE_NAME ?=simclr

container:
	docker build -t $(IMAGE_NAME) --build-arg PARENT=${IMAGE_PARENT} .

fashion_labels:
	docker run --rm -p 8501:8501 -u `id -u`:`id -g` \
	-v ${LABELS_DIR}:/label_files -v ${IMAGES_DIR}:/images -v ${OUTPUT_DIR}:/output \
	$(IMAGE_NAME) streamlit run fashion_labels.py \
	/label_files/mapping_v4.csv /label_files/all_data_v4.csv /images /output

# create npz files for train and test
# fashion_pts_codes.csv is a file with PT code to name map
# 20191025/golden_data_ids.pickle is a pickle file with golden data used for classification model training
# images for 6361 product types are downloaed to haz15647727511.cloud.wal-mart.com with train/test splits:
pretrain_dataset:
	docker run --rm -u `id -u`:`id -g` -v ${IMAGES_DIR}:/mnt/data/img \
	-v ${DATA_DIR}:/data -w /data $(IMAGE_NAME) python -u pretrain_dataset.py \
	-p fashion_pts_codes.csv -g 20191025/golden_data_ids.pickle -d /mnt/data/img/train

tensorboard:
	docker run --rm -d -v ${MODEL_DIR}:/data -p 6006:6006 \
	${IMAGE_NAME} tensorboard --logdir /data

# pretrain or finetune docker on a single GPU
simclr:
	docker run -it -u `id -u`:`id -g` -v ${DATA_DIR}:/data --runtime=nvidia \
	--shm-size=1g --ulimit memlock=-1 -e CUDA_VISIBLE_DEVICES=0 \
	${IMAGE_NAME} /bin/bash

# inside the docker
# see also run_pretrain.sh
run_pretrain:
	python run_pretrain.py --train_mode=pretrain \
  --train_batch_size=4 --train_epochs=100 \
  --learning_rate=1.0 --weight_decay=1e-6 --temperature=0.5 \
  --image_size=224 --eval_split=test --resnet_depth=50 --width_multiplier=4 \
  --use_blur=False --color_jitter_strength=0.5 \
  --data_dir=${DATA_DIR} --model_dir=${MODEL_DIR} --use_tpu=False

# see also run_finetune.sh
run_finetune:
	python run_finetune.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.05 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=256 --warmup_epochs=0 \
  --image_size=224 --eval_split=test --resnet_depth=50 --width_multiplier=4 \
  --data_dir=${DATA_DIR} --checkpoint=${CHECKPOINT} --model_dir=${MODEL_DIR} --use_tpu=False
