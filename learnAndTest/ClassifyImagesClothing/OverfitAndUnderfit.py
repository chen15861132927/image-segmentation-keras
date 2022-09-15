#region import
import os
# 导入 random(随机数) 模块
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt

import numpy as np
import pathlib
import shutil
import tempfile
print(tf.__version__)
#endregion
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:],1)
    return features, label
FEATURES = 28

#logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
#shutil.rmtree(logdir, ignore_errors=True)
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

packed_ds = ds.batch(10000).map(pack_row).unbatch()
for features,label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins = 101)



