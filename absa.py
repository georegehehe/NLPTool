#coding=utf-8
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel, BertConfig,  TFBertForTokenClassification
import tensorflow as tf
import pandas as pd
from official.nlp import optimization
import tensorflow_text as text
import numpy as np
import re
import json
#val_loss: 0.1221 - val_sparse_categorical_accuracy: 0.9562
# entity_only: val_loss: 0.0708 - val_sparse_categorical_accuracy: 0.9734
config = BertConfig.from_pretrained("hfl/chinese-macbert-base")
tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-macbert-base")
max_len = 200
BATCH_SIZE = 8
num_labels = 3
config.num_labels = num_labels
model = TFBertForTokenClassification.from_pretrained("hfl/chinese-macbert-base", config=config)
#model = TFBertForTokenClassification.from_pretrained("Weights/macbert_pretrained")
#data_path = "./Data-3-categories/
epoch = 2
num_tokens = []
emotion_dict = {-1: 'NEG', 0: 'NORM', 1: 'POS'}
emotion_dict_reverse = {'NEG': -1,'NORM': 0,'POS': 1}
#build a dictionary that transforms string lables into numerical labels
#Based on the classical BIO tagging for NER related tasks, my original idea was to
#combine sentiment analysis with NER tagging by assigning different labels to tokens
#according to their sentiment polarity.
#E.g. the sentence '我喜欢苹果但讨厌橙子' will be labeled as 'O O O B-POS I-POS O B-NEG I-NEG'
tag2id = {'O':0,'B-NEG': 1, 'I-NEG': 2, 'B-NORM': 3, 'I-NORM': 4, 'B-POS':5, 'I-POS':6}
#fpr entity_only
#tag2id = {'O':0,'B': 1, 'I': 2}
# Cut sentences
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")
# select and combine sentences into a string with a given maximum character restriction.
# As a common practice, sentences from the beginning and towards the end of the lists are prioritized, assuming that
# they would capture most meaning from the original text
def add_sentence(head,lst, limit):
    if not lst:
        return ""
    else:
        if head:
            if limit - len(lst[0]) < 0:
                return ""
            else:
                return lst[0] + add_sentence(False,lst[1:], limit - len(lst[0]))
        else:
            if limit - len(lst[-1]) < 0:
                return ""
            else:
                return add_sentence(True, lst[:-1], limit-len(lst[-1])) + lst[-1]

def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 max_length=max_len,
                                 padding='max_length',
                                 return_attention_mask=True,
                                 truncation=True,
                                 return_offsets_mapping=True
                                 )

def map_example_to_dict(input_ids, token_type_ids, attention_mask, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_mask,
           }, label
# from https: // blog.csdn.net / qq_36287702 / article / details / 123604898
# create a mapping between characters and token_spans, so that labels can be correctly assigned to
# each token given the character's index
def create_char2tok(mapping, text_len):
    char2tok_span = [[-1, -1] for _ in range(text_len)]  # [-1, -1] is whitespace
    for tok_ind, char_sp in enumerate(mapping):
        for char_ind in range(char_sp[0], char_sp[1]):
            tok_sp = char2tok_span[char_ind]
            # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
            if tok_sp[0] == -1:
                tok_sp[0] = tok_ind
            tok_sp[1] = tok_ind + 1
    return char2tok_span

# the main function for generating input data
def encode_example(examples, entity_lst, tag2id):
    global num_tokens
    input_ids_list = []
    token_type_ids_list = []
    attention_maks_list = []
    tag_list = []
    for sentence, tag in zip(examples, entity_lst):
        bert_input = convert_example_to_feature(sentence)
        #num_tokens.append(len(sentence))
        input_ids_list.append(bert_input["input_ids"])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_maks_list.append(bert_input['attention_mask'])
        offset_mapping = bert_input['offset_mapping']
        char2tok_span = create_char2tok(offset_mapping,len(sentence))
        inter_tag = [tag2id['O']] * len(bert_input["input_ids"])
        temp_tag = []
        # find all occurrences of the entities from the input sentence
        for entity, emotion in tag:
            temp_tag += [(m.start(),emotion, len(entity)) for m in re.finditer(re.escape(entity), sentence)]
        for index, emotion, entity_len in temp_tag:
            span_lst = char2tok_span[index:index+entity_len]
            #start index of token entity
            start = span_lst[0][0]
            #end index of token entity
            end = span_lst[-1][1]
            # change the labels according to the emotion of the entity
            inter_tag[span_lst[0][0]] = tag2id['B-' + emotion_dict[emotion]]
            for i in range(start+1, end, 1):
                inter_tag[i] = tag2id['I-' + emotion_dict[emotion]]
        tag_list.append(inter_tag)
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, token_type_ids_list, attention_maks_list, tag_list)).map(map_example_to_dict)

# data preprocess
# data source: 2019搜狐校园算法大赛
def txt_to_list(path):
    with open(path, 'rb') as data_file:
        json_objects = data_file.readlines()
    content_lst = []
    sent_len = []
    entity_lst = []
    for obj in json_objects:
        temp_dict = json.loads(obj.decode('utf8'))
        temp_str = temp_dict['content']
        temp_str = re.sub('\s+', '', temp_str)
        temp_entity = temp_dict['coreEntityEmotions']
        temp_entity_lst = [(ett['entity'].replace(u'\u200b',''), emotion_dict_reverse[ett['emotion']]) for ett in temp_entity]
        entity_lst.append(temp_entity_lst)
        # to reduce the size of the input sentences, only fetch the sentences that contain the entity
        untruncated_lst = []
        for sent in cut_sent(temp_str):
            if any(entity[0] in sent for entity in temp_entity_lst):
                untruncated_lst.append(sent)
        # further cap the input size to max_len while keeping full sentences
        result_str = add_sentence(True, untruncated_lst, max_len)
        sent_len.append(len(result_str))
        content_lst.append(result_str)
    #85% training, 15% validation
    content_train, content_val, entity_train, entity_val = train_test_split(content_lst, entity_lst, test_size=0.15, shuffle=True)
    num_tokens = np.array(sent_len)
    print(np.median(num_tokens))
    print(np.mean(num_tokens))
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    print(max_tokens)
    return content_train, content_val, entity_train, entity_val

train_lst_x, val_lst_x, train_lst_y, val_lst_y = txt_to_list('data_v2/coreEntityEmotion_train.txt')
train = encode_example(train_lst_x, train_lst_y, tag2id=tag2id).shuffle(len(train_lst_x)).batch(BATCH_SIZE).prefetch(1)
val = encode_example(val_lst_x, val_lst_y, tag2id=tag2id).batch(BATCH_SIZE).prefetch(1)
# check input data
input_ids_batch,label_batch = next(iter(train.take(5)))
print(input_ids_batch)
print(type(input_ids_batch))
for label in label_batch:
   print(label)

# redefined metrics and loss function with masking applied--currently not used due to unexpected errors
def sparse_categorical_accuracy_masked(y_true, y_pred):
    mask_value = -100
    active_loss = tf.reshape(y_true, (-1,)) != mask_value
    reduced_logits = tf.boolean_mask(tf.reshape(y_pred, (-1, max_len)), active_loss)
    y_true = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
    reduced_logits = tf.cast(tf.argmax(reduced_logits, axis=-1), tf.keras.backend.floatx())
    equality = tf.equal(y_true, reduced_logits)
    return tf.reduce_mean(tf.cast(equality, tf.keras.backend.floatx()))


def sparse_crossentropy_masked(y_true, y_pred):
    mask_value = -100
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, mask_value))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, mask_value))
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true_masked,
                                                                          y_pred_masked))
# define optimizer, early stopping, loss function and metrics =
step_per_epoch = tf.data.experimental.cardinality(train).numpy()
num_train_steps = step_per_epoch * epoch
num_warmup_steps = int(0.1 * num_train_steps)
init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
metrics = tf.metrics.SparseCategoricalAccuracy()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#model.layers[-1].activation = tf.keras.activations.softmax
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=metrics)
tf.keras.backend.clear_session()
model.summary()
model.fit(train, epochs=epoch, verbose=1, validation_data=val, callbacks=[callback])

#saving the models, only save_pretrained is necessary
#model.save('Bert-sent-NER_macbert', include_optimizer=False)
model.save_weights('./Weights/macbert_ckpt_entity_only')
model.save_pretrained('./Pretrained/macbert_pretrained_entity_only')

# test model accuracy---currently no testing dataset is available
#model_loss, model_acc = model.evaluate(test, verbose=1)
#print("The accuracy of the model on the test set is: ", model_acc)
#print("Loss:", model_loss)



# for MSRA dataset, not used in this model
def txt_to_csv(path):
    sent_lst = []
    label_lst = []
    with open(path, 'rb') as f:
        lines = f.readlines()
    sent = ""
    labels = []
    for line in lines:
        temp = re.sub("\s+", "", line.decode())
        if temp:
            sent += line.decode()[0]
            labels.append(temp[1:])
        else:
            sent_lst.append(sent)
            label_lst.append(labels)
            sent = ""
            labels = []
    return {"review": sent_lst, "label": label_lst}

#train_dict = txt_to_csv('Data_MSRA/msra_train_bio.txt')
#test_dict = txt_to_csv('Data_MSRA/msra_test_bio.txt')
