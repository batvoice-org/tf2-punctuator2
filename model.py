import tensorflow as tf

class BidirectionalEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(BidirectionalEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.enc_units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform'),
            merge_mode='concat')

    def call(self, x, initial_state):
        x = self.embedding(x)
        output, last_fwd_state, last_bkwd_state = self.bigru(x, initial_state=initial_state)
        return output, [last_fwd_state, last_bkwd_state]

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), ]


# refactored
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_t_m, H):
        # s_t_m: encoder state at time t - 1: initialized as the last hidden state of the encoder
        # H: (h_1, ..., h_T) (all encoder states)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(s_t_m, 1)

        score = self.V(tf.nn.tanh(
            self.W1(H) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * H
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# following notation from Tilk & Alumae paper (2016)
# var_t: var at time step t
# var_t_m: var at time step t-1
class PunctDecoder(tf.keras.Model):
    def __init__(self, vocab_size, n_units, batch_sz):
        super(PunctDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.n_units = n_units

        self.gru = tf.keras.layers.GRU(self.n_units,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.Wf_a = tf.keras.layers.Dense(n_units)
        self.Wf_f = tf.keras.layers.Dense(n_units)
        self.Wf_h = tf.keras.layers.Dense(n_units)
        self.Wy = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.n_units)
        self.s_t_m = None

    def call(self, h_t, s_t_m, H, initial_state, return_weights=False):

        context_vector, att_weights = self.attention(s_t_m, H)
        s_t, _ = self.gru(tf.expand_dims(h_t, 1), initial_state=initial_state)

        projected_context = self.Wf_a(context_vector)
        fusion_weights = tf.nn.sigmoid(self.Wf_f(projected_context)
                                       + self.Wf_h(s_t))

        f_t = tf.math.multiply(projected_context, fusion_weights) + s_t

        y_t = self.Wy(f_t)
        if return_weights:
            return y_t, s_t, att_weights
        return y_t, s_t

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.n_units))

    def get_hidden_state(self):
        return self.s_t
