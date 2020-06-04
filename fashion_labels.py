import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.title("Fashion Labels")


@st.cache
def get_labels(label_name_file, add_value=False):
    """
        get list of labels and position_by_attriubtes dict

    Args:
        label_name_file: path to the label mapping file

    Returns:
        label_names: list of label names
        positions_by_attributes: {attributes/values: pos}

    """
    label_names = []
    value_pos = 0

    # position of all values and attributes
    positions_by_attributes = {"all": []}

    with open(label_name_file, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            attribute_name = row[1]
            value_name = row[2]

            if attribute_name and not attribute_name.startswith('has'):
                if attribute_name not in positions_by_attributes:
                    positions_by_attributes[attribute_name] = []
                positions_by_attributes[attribute_name].append(value_pos)
                positions_by_attributes["all"].append(value_pos)
                if add_value:
                    positions_by_attributes['{}@{}'.format(attribute_name, value_name)] = [value_pos]
            value = row[2]
            label_name = '{}@{}'.format(attribute_name, value)
            label_names.append(label_name)

            value_pos += 1

    return np.array(label_names), positions_by_attributes


@st.cache
def get_attributes(labels_file, attr_value_positions):
    sample_rows = []
    with open(labels_file, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in reader:
            np_values = np.array(row[3:])[attr_value_positions]
            uniques, indices = np.unique(np_values, return_index=True)
            if '1' in uniques:
                sample_rows.append([row[0], indices[uniques.tolist().index('1')], row[1]])
    return sample_rows


def main(argv):
    np_label_names, positions_by_attributes = get_labels(argv[1])

    attr_list = sorted(list(positions_by_attributes.keys()))
    selected_attr = st.sidebar.selectbox('Select attribute', attr_list)
    num_items = st.sidebar.slider('Select number of items', 10, 1000, 100)

    attr_value_positions = positions_by_attributes[selected_attr]
    st.write(np_label_names[attr_value_positions])

    attr_values = get_attributes(argv[2], attr_value_positions)

    image_dir = argv[3]
    output_dir = argv[4]

    if len(attr_values) == 0:
        st.write('No result')
    else:
        urls = [t[0] for t in attr_values]
        values = [np_label_names[attr_value_positions][t[1]] for t in attr_values]
        values = [v.split('@')[1] for v in values]

        values_df = pd.DataFrame({'url': urls, 'value': values})
        uniques, counts = np.unique(values_df['value'], return_counts=True)
        value_stat_df = pd.DataFrame({'unique': uniques.tolist(), 'count': counts.tolist()})
        st.write(value_stat_df)

        st.image(image=urls[:num_items], caption=values[:num_items], width=100)

        if st.sidebar.button('Create NPZ Files'):
            start_time = time.time()
            image_data_list, label_list, url_list = [], [], []
            counter = 0
            for row in attr_values:
                filename = '%s/%s' % (image_dir, row[2])
                try:
                    image_data = tf.gfile.FastGFile(filename, 'rb').read()
                    image_data_list.append(image_data)
                    label_list.append(row[1])
                    url_list.append(row[0])

                    counter += 1
                except:
                    st.write('%s error' % filename)
                    if os.path.isfile(filename):
                        st.write('file %s exists' % filename)
                    else:
                        st.write('file %s does not exist' % filename)

                if len(image_data_list) == 1000:
                    st.write('Writing %s/%d.npz' % (output_dir, counter / 1000))
                    np.savez_compressed('%s/%d.npz' % (output_dir, counter / 1000),
                                        image=image_data_list,
                                        label=label_list,
                                        url=url_list)
                    image_data_list, label_list, url_list = [], [], []

            np.savez_compressed('%s/%d.npz' % (output_dir, counter / 1000 + 1),
                                image=image_data_list,
                                label=label_list,
                                url=url_list)

            st.write('Time taken %.2f' % (time.time() - start_time))


main(sys.argv)
