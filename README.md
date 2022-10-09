# NLP Tool
# Sentiment-Analysis

Using tensor_hub BERT model and its corresponding preprocessor to achieve the DNN based sentiment analysis of Chinese sentences.

## Requirements

```bash
pip install tensorflow==2.8
pip install tensor_hub
pip install -q -U "tensorflow-text==2.8.*"
pip install -q tf-models-official==2.7.0
```
make sure to correctly install Nvidia Cuda for running tensorflow on GPU

## Files
1. Bert-sent-analysis.h5 saves model weights/assets (not pushed to repo due to large size)
2. quote_classifier.py uses the saved model to analyze sentences from quote.xlsx 
3. quote_with_labels.xlsx/quote_with_labels_v2.xlsx are outputs from the previous script. They differ due to the data that they've been trained on
4. test_accuracy.py tests the model accuracy
5. Bert_classifier.py builds the Bert-based model
6. bert_chinese_model contains tensor_hub layer files (the preprocessor and the pre fine-tuned Bert model, also not pushed)
7. Data directory contains the training/testing files
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
2. Seems to miss-interpret negation and transition words
3. Mediocre performance on the actual quotes that were to be analyzed
## Sources
https://www.tensorflow.org/text/tutorials/classify_text_with_bert

https://tfhub.dev/tensorflow/bert_zh_preprocess/3

https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4

# Named Entity Recognition

Using chinese_macbert_base model and its corresponding tokenizer to achieve both Named Entity Recognition on Chinese sentences and quasi Aspect Based Sentiment Analysis(ABSA).

## Requirements

```bash
pip install tensorflow==2.8
pip install -q -U "tensorflow-text==2.8.*"
pip install -q tf-models-official==2.7.0
pip install transformers
pip install pandas
pip install numpy
pip install scikit-learn
```
make sure to correctly install Nvidia Cuda for running tensorflow on GPU

## Files
1. absa.py trains the macbert-base model for NER tasks. Currently the code will output an ABSA model, but simple modifications allows the model to onnly focus on entity-recognition tasks
2. quote_classifier_ner.py uses the saved model to analyze sentences from quote.xlsx 
3. quote_with_labels_entity_only.xlsx/quote_with_labels_entity+emotion.xlsx are outputs from the previous script. The former only marks the entities within the sentences while the latter includes the sentiment of the entities.

## How It Works?

To complete most NER tasks, the model is fine-tuned in a way that reads in a series of tokens and outputs a series of labels corresponding to each token.
This program relies on the same insight. Based on the classical BIO tagging for NER related tasks, the idea was to combine sentiment analysis with NER tagging by assigning different labels to tokens according to their sentiment polarity. E.g. the sentence '我喜欢苹果但讨厌橙子'(meaning 'I like apples but hate oranges') will be labeled as 'O O O B-POS I-POS O B-NEG I-NEG', because the word 'apple' is clearly assigned with a positive sentiment and 'orange' the contrary. It is hoped that by exposing the model to training sets under such labeling mechanism, the model will pick up the entity and the emotion simultaneously. 

## pros
1. high accuracy: 95.6% accuracy on the validation set for ABSA and 97.3% for entity only
2. BERT take the context of the whole sentence into account
## cons
1. expensive training time and memory usage
2. Sometimes impossible labeling would appear (e.g. having an I label without a preceding B label, or having a padded token assuming a non O label)
3. Works with much less accuracy for actual quotes, especially for ABSA.
## Next Step
1. A CRF layer can be added after the bert model to classify out the impossible cases
2. Incorporating a double-verification system in the model; adding a sequential model that completes NER first then Sentiment Analsysis and combine the results with the current model to derive the final output
3. Implementing customized loss functions to deal with padded tokens specially




