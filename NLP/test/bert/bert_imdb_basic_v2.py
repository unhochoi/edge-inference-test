# Install the required package
# !pip install bert-for-tf2
# !pip install tensorflow_hub

# Import modules
import os
import re
import bert
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm
from tensorflow.keras.models import load_model

print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = "positive"
  neg_df["polarity"] = "negative"
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)
  
  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
  
  return train_df, test_df

def create_tonkenizer(bert_layer):
    """Instantiate Tokenizer with vocab"""
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy() 
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
    print("Vocab size:", len(tokenizer.vocab))
    return tokenizer

def get_ids(tokens, tokenizer, MAX_SEQ_LEN):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return input_ids

def get_masks(tokens, MAX_SEQ_LEN):
    """Masks: 1 for real tokens and 0 for paddings"""
    return [1] * len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))

def get_segments(tokens, MAX_SEQ_LEN):
    """Segments: 0 for the first sequence, 1 for the second"""  
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))

def create_single_input(sentence, tokenizer, max_len):
    """Create an input from a sentence"""
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:max_len] # max_len = MAX_SEQ_LEN - 2, why -2 ? ans: reserved for [CLS] & [SEP]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    return get_ids(stokens, tokenizer, max_len+2), get_masks(stokens, max_len+2), get_segments(stokens, max_len+2)
  
def convert_sentences_to_features(sentences, tokenizer, MAX_SEQ_LEN):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences, position=0, leave=True):
      ids, masks, segments = create_single_input(sentence, tokenizer, MAX_SEQ_LEN-2) # why -2 ? ans: reserved for [CLS] & [SEP]
      input_ids.append(ids)
      input_masks.append(masks)
      input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]

def nlp_model(bert_base):
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=bert_base, trainable=True)  
    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")           
    input_masks = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")       
    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")

    inputs = [input_ids, input_masks, input_segments] # BERT inputs
    pooled_output, sequence_output = bert_layer(inputs) # BERT outputs
    
    x = Dense(units=768, activation='relu')(pooled_output) # hidden layer 
    x = Dropout(0.15)(x) 
    outputs = Dense(2, activation="softmax")(x) # output layer

    model = Model(inputs=inputs, outputs=outputs)
    return model


# hyper-parameters
MAX_SEQ_LEN = 500

# model construction (we construct model first inorder to use bert_layer's tokenizer)
bert_base = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
model = nlp_model(bert_base) 
model.summary()

train_df, test_df = download_and_load_datasets()
df = pd.concat([train_df, test_df])
df = df.drop("sentiment", axis=1)
df.rename(columns = {'sentence': 'review', 'polarity' : 'sentiment'}, inplace=True)

# Create examples for training and testing
df = df.sample(frac=1) # Shuffle the dataset
# we would like to use bert tokenizer; therefore, chech model.summary() and find the index of bert_layer
tokenizer = create_tonkenizer(model.layers[3])

# create training data and testing data
x_train = convert_sentences_to_features(df['review'][:40000], tokenizer, MAX_SEQ_LEN)
x_valid = convert_sentences_to_features(df['review'][40000:45000], tokenizer, MAX_SEQ_LEN)
x_test = convert_sentences_to_features(df['review'][45000:], tokenizer, MAX_SEQ_LEN)
df['sentiment'].replace('positive', 1., inplace=True)
df['sentiment'].replace('negative', 0., inplace=True)
one_hot_encoded = to_categorical(df['sentiment'].values)
y_train = one_hot_encoded[:40000]
y_valid = one_hot_encoded[40000:45000]
y_test =  one_hot_encoded[45000:]

BATCH_SIZE = 8
EPOCHS = 1

# use adam optimizer to minimize the categorical_crossentropy loss
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# fit the data to the model
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# save the trained model
model.save('./bert_imdb.h5')

# load the pretrained nlp_model
imdb_model = load_model('./bert_imdb.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# predict on test dataset
y_pred = np.argmax(model.predict(x_test), axis=1)

# 모든 데이터에 대한 실제라벨과 예측라벨을 비교한 뒤, 정확도 계산
real_labels = np.array(np.argmax(y_test, axis=1))
pred_labels = np.array(y_pred)

accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
print(accuracy)
