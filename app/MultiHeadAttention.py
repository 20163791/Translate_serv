import tensorflow as tf

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super().__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, mask=None):
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            
            if mask is not None:
                score *= mask
                score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)
            alignment = tf.nn.softmax(score, axis=2)
            head = tf.matmul(alignment, self.wv[i](value))
            heads.append(head)
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        return heads