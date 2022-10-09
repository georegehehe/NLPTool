import os
import shutil
import random
import pandas
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from official.nlp import optimization

print(tf.__version__)
import matplotlib.pyplot as plt

test_dir = "./Data/test.csv"
train_dir = "./Data/train.csv"
preprocess_tfhub = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"
encoder_path = "./bert_chinese_model"
BATCH_SIZE = 128

test_feature_df = pd.read_csv(test_dir)
test_len = len(test_feature_df)
test_label_df = test_feature_df.pop('label')
train_feature_df = pd.read_csv(train_dir)
train_len = len(train_feature_df)
train_label_df = train_feature_df.pop('label')
train_dataset = tf.data.Dataset.from_tensor_slices((train_feature_df.values, train_label_df.values)).shuffle(train_len//4).batch(BATCH_SIZE).prefetch(1)

test_dataset = tf.data.Dataset.from_tensor_slices((test_feature_df.values, test_label_df.values)).shuffle(test_len//4).batch(BATCH_SIZE).prefetch(1)

#feature_batch, label_batch = next(iter(train_dataset.take(5)))
#for label in label_batch:
 #   print(label)
def build_classfier_model():
    text_input = tf.keras.layers.Input(shape=(),dtype=tf.string, name='text')
    preprocessor = hub.KerasLayer(
        preprocess_tfhub,
        name='preprocessing'
    )
    encoder_inputs = preprocessor(text_input)
    print("test1")
    encoder = hub.KerasLayer(
        encoder_path,
        trainable=True,
        name='BERT_encoder'
    )
    print("test1")
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None)(net)
    return tf.keras.Model(text_input, net)

model = build_classfier_model()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

epochs = 5
step_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
num_train_steps = step_per_epoch * epochs
num_warmup_steps = int(0.1 * num_train_steps)
init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
history = model.fit(x=train_dataset,validation_data=test_dataset,epochs=epochs)
"""
df = pandas.read_csv("./Data/weibo_senti_100k.csv")
limit = 10000
df = df.sample(frac=0.1)
label = df.label
features = df.review
SEED = 42

feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size=0.2, shuffle=True)
out_train = dict()
out_test = dict()
out_train['review'] = feature_train
out_train['label'] = label_train
out_test['review'] = feature_test
out_test['label'] = label_test
pd.DataFrame(out_train).to_csv("Data/train.csv", index=False)
pd.DataFrame(out_test).to_csv("Data/test.csv", index=False)
"""




