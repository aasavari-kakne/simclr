"""TODO(pretrain_dataset): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import tensorflow_datasets.public_api as tfds
import hashlib
import json

# TODO(pretrain_dataset): BibTeX citation
_CITATION = """
"""

# TODO(pretrain_dataset):
_DESCRIPTION = """
"""


PREFIX = '/app/furniture-style/'


setting_folder = '/Users/amagnan/Documents/workspace/vision/xplore_2019/data'

train_file = os.path.join(setting_folder,'train.csv')
val_file = os.path.join(setting_folder, 'validation.csv')
test_file = os.path.join(setting_folder,'test.csv')
setting_file = os.path.join(setting_folder,'settings.json')

# this is the absolute path of the folder containing the images. Images are under data/images
# change this to meet the settings in your syste,
image_folder = '/Users/amagnan/Documents/workspace/vision/xplore_2019/data/images'




class FinetuneDatasetInfo(tfds.core.DatasetInfo):
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


class FinetuneDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

    with open(setting_file, 'r') as f:
        SETTINGS = json.load(f)
    NUM_CLASSES = len(SETTINGS.get('style').get('code_to_name'))
    print('there are {}  classes'.format(NUM_CLASSES))
    def _info(self):
        # TODO(pretrain_dataset): Specifies the tfds.core.DatasetInfo object
        return FinetuneDatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(num_classes=self.NUM_CLASSES),
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



        product_code_to_name = self.SETTINGS.get('product_type').get('code_to_name')
        style_code_to_name = self.SETTINGS.get('style').get('code_to_name')

        def read_file(filename):
            out  = dict()
            with open(filename, 'r')  as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                for row in reader:
                    rel_path = row[0]
                    rel_path = rel_path[len(PREFIX):]

                    full_path = os.path.join(image_folder, rel_path)

                    assert (os.path.isfile(full_path))
                    product_type_code = row[2]
                    style_code = row[3]

                    product_type = product_code_to_name.get(product_type_code)
                    style = style_code_to_name.get(style_code)

                    style_code = int(style_code)
                    product_type_code =  int(product_type_code)

                    #yield full_path, product_type_code, product_type, style_code, style
                    #print(full_path, product_type_code, product_type, style_code, style)
                    if style not in out:
                        out[style] =  {'image_path': [], 'label': style_code}
                    out[style]['image_path'].append(full_path)
            return out

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "label_images": read_file(train_file),
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "label_images": read_file(test_file),
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


