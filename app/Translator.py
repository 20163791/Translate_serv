import tensorflow as tf

class Translator():
    def predict( tokenizer_1, tokenizer_2, encoder, decoder, test_source_text):
        print(test_source_text)
        test_source_seq = tokenizer_1.texts_to_sequences([test_source_text])
        print(test_source_seq)
        en_output = encoder(tf.constant(test_source_seq))
        de_input = tf.constant([[tokenizer_2.word_index['<start>']]], dtype=tf.int64)
        out_words = []

        while True:
            de_output = decoder(de_input, en_output)
            new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
            out_words.append(tokenizer_2.index_word[new_word.numpy()[0][0]])
            de_input = tf.concat((de_input, new_word), axis=-1)
            if out_words[-1] == '<end>' or len(out_words) >= 14:
                break

        return ' '.join(out_words)


