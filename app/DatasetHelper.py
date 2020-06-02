import tensorflow as tf
import numpy as np
import unicodedata
import re
import io


class DatasetHelper():
    def __init__(self, path_to_file):
        super().__init__()
        self.path_to_file = path_to_file
        self.tokenizer_1 = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.tokenizer_2 = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.raw_data_2_in  = None
        self.raw_data_2_out = None

    def __unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def __normalize_string(self, s):
        s = self.__unicode_to_ascii(s)
        s = re.sub(r'([!.?])', r' \1', s)
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        s = re.sub(r'\s+', r' ', s)
        return s

    def __create_dataset(self, num_examples):
        lines = io.open(self.path_to_file, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[w for w in l.split('\t')] for l in lines[:num_examples]]
        return zip(*word_pairs)

    def get_raw_data(self):
        raw_data_1, raw_data_2, trash = self.__create_dataset( 3000)
        raw_data_1, raw_data_2 = list(raw_data_1), list(raw_data_2)
        raw_data_1 = [self.__normalize_string(data) for data in raw_data_1]
        self.raw_data_2_in = ['<start> ' + self.__normalize_string(data) for data in raw_data_2] #only for DatasetHelper
        self.raw_data_2_out = [self.__normalize_string(data) + ' <end>' for data in raw_data_2]  #only for DatasetHelper
        return raw_data_1, raw_data_2

    def get_tokenized_data(self):
        raw_data_1, raw_data_2 = self.get_raw_data()
        self.tokenizer_1.fit_on_texts(raw_data_1)
        data_1 = self.tokenizer_1.texts_to_sequences(raw_data_1)
        data_1 = tf.keras.preprocessing.sequence.pad_sequences(data_1,padding='post')
        self.tokenizer_2.fit_on_texts(self.raw_data_2_in)
        self.tokenizer_2.fit_on_texts(self.raw_data_2_out)
        data_2_in = self.tokenizer_2.texts_to_sequences(self.raw_data_2_in)
        data_2_in = tf.keras.preprocessing.sequence.pad_sequences(data_2_in,padding='post')
        data_2_out = self.tokenizer_2.texts_to_sequences(self.raw_data_2_out)              #only for DatasetHelper
        data_2_out = tf.keras.preprocessing.sequence.pad_sequences(data_2_out,padding='post')
        return data_1, data_2_in, data_2_out

    def get_dataset(self, BATCH_SIZE):
        data_1, data_2_in, data_2_out = self.get_tokenized_data()
        dataset = tf.data.Dataset.from_tensor_slices(
            (data_1, data_2_in, data_2_out))
        dataset = dataset.shuffle(20).batch(BATCH_SIZE)
        return dataset

    def __positional_embedding(self, pos, model_size):
        PE = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            else:
                PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
        return PE

    def get_postional_encoding(self, MODEL_SIZE):
        data_1, data_2_in, data_2_out = self.get_tokenized_data()
        max_length = max(len(data_1[0]), len(data_2_in[0]))
        pes = []
        for i in range(max_length):
            pes.append(self.__positional_embedding(i, MODEL_SIZE))
        pes = np.concatenate(pes, axis=0)
        pes = tf.constant(pes, dtype=tf.float32)
        return pes