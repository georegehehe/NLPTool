#coding=utf-8
import re
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import os
import shutil
import random
import pandas
from tensorflow.python.ops.numpy_ops import np_config
from transformers import BertTokenizerFast, TFBertForTokenClassification, BertConfig
import jieba.posseg as pseg
import jieba
import codecs

tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-macbert-base")

tag2id = {'O':0,'B-NEG': 1, 'I-NEG': 2, 'B-NORM': 3, 'I-NORM': 4, 'B-POS':5, 'I-POS':6}
emotion_dict_reverse = {'NEG': -1,'NORM': 0,'POS': 1}
#for entity only
#tag2id = {'O':0,'B': 1, 'I':2}
id2tag = {v: k for k, v in tag2id.items()}
def tokenization(file_context):
    result = []
    words = pseg.cut(file_context)
    for word, flag in words:
        if flag in nflag and word not in stopwords and len(word)>1:
            result.append(word)
    return result

def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 max_length=len(max(quotes, key=len)) + 2,
                                 padding='max_length',
                                 truncation=True,
                                 return_offsets_mapping=True
                                 )

#convert sentences to numerical inputs
def tokenize_texts(quote_lst):
    input_ids_lst = []
    offset_mapping_lst = []
    for sent in quote_lst:
        bert_input = convert_example_to_feature(sent)
        input_ids_lst.append(bert_input['input_ids'])
        offset_mapping_lst.append(bert_input['offset_mapping'])
    return input_ids_lst, offset_mapping_lst

#converts label output into list of entities and sentiment polarity
def label2entity(lst, mapping_index, sent):
    in_word = False
    curr_index = 1
    temp_emotion = []
    emotion_lst = []
    word_lst = []
    for i, label in enumerate(lst, 1):
        if label == 'O':
            if in_word:
                #map tokens to characters using offset mapping
                word_start = offset_mapping_lst[mapping_index][curr_index:i][0][0]
                word_end = offset_mapping_lst[mapping_index][curr_index:i][-1][1]
                word_lst.append(sent[word_start:word_end])
                mean = round(sum(temp_emotion) / len(temp_emotion))
                emotion_lst.append(mean)
                in_word = False
                temp_emotion = []
        if label[0] == 'B':
            if in_word:
                word_start = offset_mapping_lst[mapping_index][curr_index:i][0][0]
                word_end = offset_mapping_lst[mapping_index][curr_index:i][-1][1]
                word_lst.append(sent[word_start:word_end])
                mean = round(sum(temp_emotion) / len(temp_emotion))
                emotion_lst.append(mean)
                temp_emotion = []
            in_word = True
            curr_index = i
            temp_emotion.append(emotion_dict_reverse[label[2:]])

        if label[0] == 'I':
            if not in_word:
                curr_index = i
                in_word = True
            temp_emotion.append(emotion_dict_reverse[label[2:]])

        if i == len(lst):
            if in_word:
                word_start = offset_mapping_lst[mapping_index][curr_index:i+1][0][0]
                word_end = offset_mapping_lst[mapping_index][curr_index:i+1][-1][1]
                word_lst.append(sent[word_start:word_end])
                mean = round(sum(temp_emotion) / len(temp_emotion))
                emotion_lst.append(mean)
                temp_emotion = []
    assert len(word_lst) == len(emotion_lst)
    return word_lst, emotion_lst

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

#for COVID model
#ht = {2:0,0:1,1:2}
#for news
#ht = {1:2,2:0,0:1}
#for NER
reloaded = TFBertForTokenClassification.from_pretrained("Pretrained/macbert_pretrained_entity_only")
quote_len = np.array(list(map(lambda x:len(x), quotes)))
print(np.mean(quote_len) + 2 * np.std(quote_len))

config = reloaded.get_config()
print(config) # returns a tuple of width, height and channels
feed_lst, offset_mapping_lst = tokenize_texts(quotes)
#softmax convert output values into probabilities
pred = reloaded.predict(feed_lst)[0]
print(pred)
results = tf.nn.softmax(pred)
print(results)
# get the list of labels for the input quotes
result_lst = []
for i in range(len(quotes)):
    temp = [id2tag[int(tf.math.argmax(results[i][j]))] for j in range(1, len(quotes[i]) + 1, 1)]
    result_lst.append(temp)

final_ent = []
final_emo = []
for i, res in enumerate(result_lst):
    temp_ent, temp_emo = label2entity(res, i, quotes[i])
    final_ent.append(temp_ent)
    final_emo.append(temp_emo)

print(final_emo)
print(final_ent)




#output as excel
out_dict = dict()
out_dict['quote'] = quotes
out_dict['label'] = final_emo
out_dict['entity'] = final_ent
out_dict['time'] = time_lst
out_dict['location'] = loc_lst
out_df = pandas.DataFrame(data=out_dict)
out_df.to_excel('quote_with_labels_entity_only.xlsx')