import re
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
import shutil
import random
import pandas
from tensorflow.python.ops.numpy_ops import np_config
import jieba.posseg as pseg
import jieba
import codecs

# Cut sentences

def tokenization(file_context):
    result = []
    words = pseg.cut(file_context)
    for word, flag in words:
        if flag in nflag and word not in stopwords and len(word)>1:
            result.append(word)
    return result


np_config.enable_numpy_behavior()

# Require stop words dictionary
stop_words = './Stop_Words_Full.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]
#目前只包含了普通名词，如有需要可以添加 https://github.com/fxsjy/jieba
nflag = ['n']

df = pandas.read_excel("quote.xlsx")
quotes = df.iloc[:, 1].tolist()
time_lst = df.iloc[:, 2].tolist()
loc_lst = df.iloc[:, 3].tolist()
noun_lst = []
i = 0
#convert the string representation of the whole list list into a list of strings
while i < len(quotes):
    if quotes[i][0] == '[':
        insert_lst = re.findall('\'(.*?)\'', quotes[i])
        for segment in insert_lst:
            quotes.insert(i,segment)
            time_lst.insert(i, time_lst[i])
            loc_lst.insert(i, loc_lst[i])
            i += 1
        quotes.pop(i)
        time_lst.pop(i)
        loc_lst.pop(i)
    else:
        i += 1

for quote in quotes:
    temp = ""
    token_lst = tokenization(quote)
    for x, word in enumerate(token_lst):
        temp += word
        if x != len(token_lst) - 1:
            temp += "、"
    noun_lst.append(temp)


reloaded = tf.keras.models.load_model("./Bert-sent-analysis.h5", custom_objects={'KerasLayer': hub.KerasLayer})
#sigmoid function is required to turn the output of the model to values between 1 and 0 
results = tf.sigmoid(reloaded.predict(quotes))
result_lst = [int(round(results[i][0])) for i in range(len(quotes))]
out_dict = dict()
out_dict['quote'] = quotes
out_dict['label'] = result_lst
out_dict['noun'] = noun_lst
out_dict['time'] = time_lst
out_dict['location'] = loc_lst
out_df = pandas.DataFrame(data=out_dict)
out_df.to_excel('quote_with_labels.xlsx')
