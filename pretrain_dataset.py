"""TODO(pretrain_dataset): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import io
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import sys
import hashlib

# TODO(pretrain_dataset): BibTeX citation
_CITATION = """
"""

# TODO(pretrain_dataset):
_DESCRIPTION = """
"""

NUM_CLASSES  = 16
FOLDS = 10

class PretrainDatasetInfo(tfds.core.DatasetInfo):
    def initialize_from_bucket(self):
        print("calling initialize_from_bucket")
        pass





# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

md5 = hashlib.md5()

def file_hash(filename):
    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


class PretrainDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        # TODO(pretrain_dataset): Specifies the tfds.core.DatasetInfo object
        return PretrainDatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                "label_name": tfds.features.Text()
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("image", "label"),
            # Homepage of the dataset for documentation
            #homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):


        def find_images(folder, is_train):

            seen_hashes = dict()
            all_files = set()
            for subdir, dirs, files in os.walk(folder):
                for file in files:
                    hash_id = hash(file) % FOLDS
                    is_train_image = hash_id != 0
                    if (is_train_image and is_train) or (not is_train_image and not is_train):
                        file = os.path.join(subdir, file)
                        if file.lower().endswith('jpg') or file.lower().endswith('jpeg'):
                            if os.path.isfile(file):
                                # fh = file_hash(file)
                                # if fh not in seen_hashes:
                                #     seen_hashes[fh] = file
                                # else:
                                #     raise Exception('file: {} and {} have same hash'.format(file, seen_hashes[fh]))
                                all_files.add(file)
            tf.logging.info('there are {} files, is train: {}'.format(len(all_files), is_train))
            all_files =  list(all_files)
            #all_files = all_files[:10000]

            size =   int(len(all_files)/ NUM_CLASSES)

            out = dict()
            for i in range(NUM_CLASSES):
                out["label_{}".format(i)] = {'image_path': all_files[size * i: size * (i+1)], 'label': i}

            return  out

        folder = os.path.join(self._original_state['data_dir'], 'raw_images')
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "label_images": find_images(folder, is_train=True),
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "label_images": find_images(folder, is_train=False),
                }
            ),
        ]

    def _generate_examples(self, label_images):
        for label_name, image_info in label_images.items():
            for image_path in image_info['image_path']:
                key = "%s/%s" % (label_name, image_path)
                yield key, {
                    "image": image_path,
                    "label_name": label_name,
                    "label": image_info['label'],
                }


