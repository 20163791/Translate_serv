from Encoder import Encoder
from Decoder import Decoder
from DatasetHelper import DatasetHelper
from Trainer import Trainer
from Translator import Translator
from flask import Flask, jsonify
import time
import os
import tensorflow as tf

BATCH_SIZE = 5
MODEL_SIZE = 128
H = 2
NUM_LAYERS = 2
NUM_EPOCHS = 1

start_time = time.time()
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

dir_path = os.path.dirname(os.path.realpath(__file__))
path_to_file = dir_path+"/Training_data/lit.txt"

dsHelper = DatasetHelper(path_to_file)
dataset = dsHelper.get_dataset(BATCH_SIZE)
vocab_size_1 = len(dsHelper.tokenizer_1.word_index) + 1
vocab_size_2 = len(dsHelper.tokenizer_2.word_index) + 1
pes = dsHelper.get_postional_encoding(MODEL_SIZE)

encoder = Encoder(vocab_size_1, MODEL_SIZE, NUM_LAYERS, pes, H)
decoder = Decoder(vocab_size_2, MODEL_SIZE, NUM_LAYERS, pes, H)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

app = Flask(__name__)
@app.route('/')
def index():
    rez = 'Hello'
    response = jsonify({'rez':rez})
    response.status_code = 202 
    return response

@app.route('/train/<int:NUM_EPOCHS>', methods=['GET'])
def train(NUM_EPOCHS):
    Trainer.train(NUM_EPOCHS, dataset, checkpoint, checkpoint_prefix, encoder, decoder, optimizer)
    rez = f'trained for {NUM_EPOCHS}'
    response = jsonify(rez)
    response.status_code = 202 
    return response

@app.route('/translate/<input_text>', methods=['GET'])
def translate(input_text):
    output_text = Translator.predict(dsHelper.tokenizer_1, dsHelper.tokenizer_2, encoder, decoder, input_text)
    response = jsonify({'output_text':output_text})
    response.status_code = 202 
    return response