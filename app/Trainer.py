import tensorflow as tf

class Trainer():
    
    def get_loss(targets, logits):
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = crossentropy(targets, logits, sample_weight=mask)

        return loss

    @tf.function
    def train_step(source_seq, target_seq_in, target_seq_out, encoder, decoder, optimizer):
        with tf.GradientTape() as tape:
            padding_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
            padding_mask = tf.expand_dims(padding_mask, axis=1)
            encoder_output = encoder(source_seq, padding_mask)
        
            decoder_output = decoder(target_seq_in, encoder_output, padding_mask)

            loss = Trainer.get_loss(target_seq_out, decoder_output)

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def train(NUM_EPOCHS, dataset, checkpoint, checkpoint_prefix, encoder, decoder, optimizer):
        for e in range(NUM_EPOCHS):
            for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
                loss = Trainer.train_step(source_seq, target_seq_in, target_seq_out, encoder, decoder, optimizer)

            print('Epoch {} Loss {:.4f}'.format(
                  e + 1, loss.numpy()))

            if (e + 1) % 2 == 0:
                    checkpoint.save(file_prefix = checkpoint_prefix)

