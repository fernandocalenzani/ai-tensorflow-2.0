from __future__ import print_function

import tempfile

import apache_beam.io.iobase  # Adicionado novo import
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
from tensorflow_transform.tf_metadata import (dataset_metadata, dataset_schema,
                                              schema_utils)

dataset = pd.read_csv("pollution-small.csv")

features = dataset.drop("Date", axis=1)

dict_features = list(features.to_dict("index").values())

data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec({
    "no2": tf.io.FixedLenFeature([], tf.float32),
    "pm10": tf.io.FixedLenFeature([], tf.float32),
    "so2": tf.io.FixedLenFeature([], tf.float32),
    "soot": tf.io.FixedLenFeature([], tf.float32),
}))


def preprocessing_fn(inputs):
  no2 = inputs["no2"]
  pm10 = inputs["pm10"]
  so2 = inputs["so2"]
  soot = inputs["soot"]

  no2_normalized = no2 - tft.mean(no2)
  so2_normalized = so2 - tft.mean(so2)

  pm10_normalized = tft.scale_to_0_1(pm10)
  soot_normalized = tft.scale_by_min_max(soot)

  return {
      "no2_normalized": no2_normalized,
      "so2_normalized": so2_normalized,
      "pm10_normalized": pm10_normalized,
      "sott_normalized": soot_normalized
  }


def data_transform():
  with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, transform_fn = (
        (dict_features, data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

  transformed_data, transformed_metadata = transformed_dataset

  for i in range(len(transformed_data)):
    print("Initial: ", dict_features[i])
    print("Transformed: ", transformed_data[i])

    
