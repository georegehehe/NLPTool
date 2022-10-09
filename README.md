# Sentiment-Analysis

使用tensor_hub BERT模型以及相对应的preprocessor实现对中文句子的情感分析

## Requirements

```bash
pip install tensorflow==2.8
pip install tensor_hub
pip install -q -U "tensorflow-text==2.8.*"
pip install -q tf-models-official==2.7.0
```
## Files
1. Bert-sent-analysis.h5储存模型weights以及其他参数
2. quote_classifier.py使用模型对quote.xlsx文件进行分析，输出quote_with_labels.xlsx
3. test_accuracy.py测试模型准确率
4. Bert_classifier.py用来搭建模型
5. bert_chinese_model 含有tensor_hub layer属性文件
6. Data保留用来训练的文本
## Usage

```python
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
reloaded = tf.keras.models.load_model("./Bert-sent-analysis.h5", custom_objects={'KerasLayer': hub.KerasLayer})
#input_quotes = a list of strings
results = tf.sigmoid(reloaded.predict(input_quotes))
#result_lst = a list of integers corresponding to the sentiment; 1 = positive, 0 = negative
result_lst = [int(round(results[i][0])) for i in range(len(quotes))]
```
## pros
1. high accuracy: 97.8% correctness on the testing set
2. BERT take the context of the whole sentence into account
## cons
1. expensive training time and memory usage
2. Seems to miss-interpret 否定词 and 转折句 
## Sources
https://www.tensorflow.org/text/tutorials/classify_text_with_bert

https://tfhub.dev/tensorflow/bert_zh_preprocess/3

https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4