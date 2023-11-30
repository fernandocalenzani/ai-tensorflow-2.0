from __future__ import print_function

import datetime
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv

dataset = pd.read_csv(
    'app/data/TensorFlow 2.0 O Guia Completo sobre o novo TensorFlow/TensorFlow 2.0 O Guia Completo sobre o novo TensorFlow/pollution-small 1.csv')

data_l, data_c = dataset.shape

len_data_train = math.ceil(data_l*0.8)

training_data = dataset[:len_data_train]
test_data = dataset[len_data_train:]

statistics_train_data = training_data.describe()
statistics_test_data = test_data.describe()

print(statistics_train_data)
print(statistics_test_data)

train_stats = tfdv.generate_statistics_from_dataframe(dataframe=training_data)
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema)

test_stats = tfdv.generate_statistics_from_dataframe(dataframe=test_data)
anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)

test_set_copy = test_data.copy()
test_set_copy.drop("soot", axis=1, inplace=True)
test_set_copy.describe()

test_set_copy_stats = tfdv.generate_statistics_from_dataframe(
    dataframe=test_set_copy)
anomalies_new = tfdv.validate_statistics(
    statistics=test_set_copy_stats, schema=schema)
tfdv.display_anomalies(anomalies_new)

schema.default_environment.append("TRAINING")
schema.default_environment.append("SERVING")

tfdv.get_feature(schema, "soot").not_in_environment.append("SERVING")
serving_env_anomalies = tfdv.validate_statistics(
    test_set_copy_stats, schema, environment="SERVING")
tfdv.display_anomalies(serving_env_anomalies)

tfdv.write_schema_text(schema=schema, output_path="pollution_schema.pbtxt")
