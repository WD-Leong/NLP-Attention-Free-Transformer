
import numpy as np
import tensorflow as tf

# Layer Normalization. #
class LayerNorm(tf.keras.layers.Layer):
    def __init__(
        self, d_model, epsilon=1.0e-3, 
        center=True, scale=True, use_var=True):
        # center = True will return Layer Normalization, # 
        # center = False will return RMS Normalization.  #
        super(LayerNorm, self).__init__()
        self.center  = center
        self.epsilon = epsilon
        self.use_var = use_var

        if center:
            self.beta = self.add_weight(
                name="beta", shape=d_model, 
                initializer="zeros", trainable=True)
        else:
            self.beta = 0.0
        
        if scale:
            self.gamma = self.add_weight(
                name="gamma", shape=d_model, 
                initializer="ones", trainable=True)
        else:
            self.gamma = 1.0
    
    def call(self, x):
        if self.center:
            x_mean  = tf.reduce_mean(x, axis=-1, keepdims=True)
        
            if self.use_var:
                # Use standard deviation. #
                x_sigma = tf.math.sqrt(tf.reduce_mean(
                    tf.square(x - x_mean), axis=-1, keepdims=True))
            else:
                # Use mean absolute deviation. #
                x_sigma = tf.reduce_mean(
                    tf.abs(x - x_mean), axis=-1, keepdims=True)

            x_scale = tf.divide(
                x - x_mean, x_sigma + self.epsilon)
        else:
            x_sigma = tf.math.sqrt(tf.reduce_mean(
                tf.square(x), axis=-1, keepdims=True))
            x_scale = tf.divide(x, x_sigma + self.epsilon)
        
        x_output = self.gamma * x_scale + self.beta
        return x_output

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(d_model / n_heads)
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)
    
    def scaled_dot_product_attn(
        self, q, k, v, mask=None, neg_infty=-1.0e9):
        """
        For reference.
        """
        # Head dimension. #
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        lq = tf.shape(q)[2]
        lk = tf.shape(k)[2]
        
        # Multiplicative Attention. #
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale multiplicative attention mechanism. #
        attn_logits = matmul_qk * tf.math.rsqrt(dk)
        
        # Add the mask to the attention mechanism. #
        if mask is not None:
            attn_mask = (mask * neg_infty)
        else:
            attn_mask = tf.zeros([lq, lk])
        attn_logits += attn_mask
        
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_outputs = tf.matmul(attn_weights, v)
        return attn_outputs, attn_weights

    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def call(self, v, k, q, mask=None):
        q_heads = tf.nn.elu(
            self.split_heads(self.wq(q))) + 1.0
        k_heads = tf.nn.elu(
            self.split_heads(self.wk(k))) + 1.0
        v_heads = self.split_heads(self.wv(v))
        
        k_cumsum  = tf.math.cumsum(k_heads, axis=1)
        k_softmax = tf.divide(k_heads, k_cumsum)
        kv_output = tf.multiply(k_softmax, v_heads)
        
        attn_free = tf.multiply(q_heads, kv_output)
        attn_out  = self.wc(self.combine_heads(attn_free))
        return attn_out
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        return self.ffwd_2(self.ffwd_1(x))

# GPT Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, rate1=0.1, rate2=0.1):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(d_model, n_heads)
        
        self.lnorm_1 = LayerNorm(d_model, epsilon=1.0e-6)
        self.lnorm_2 = LayerNorm(d_model, epsilon=1.0e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_enc, x_pos, training=True, mask=None):
        # Apply Pre Layer Normalisation. #
        x_embed = x_enc + x_pos
        x_norm  = self.lnorm_1(x_embed)
        attn_free_out = self.attn_self(
            x_norm, x_norm, x_norm, mask=mask)
        
        attn_self_output = tf.add(
            x_embed, attn_free_out)
        attn_self_output = self.dropout_1(
            attn_self_output, training=training)
        
        # Apply Feed Forward with Pre Layer Normalisation. #
        ffwd_self_norm = self.lnorm_2(attn_self_output)
        ffwd_self_output = self.ffwd_self(ffwd_self_norm)
        
        # Residual Connection. #
        res_output = tf.add(
            attn_self_output, ffwd_self_output)
        res_output = self.dropout_2(res_output, training=training)
        return res_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        self.pos_embed = [tf.keras.layers.Embedding(
            max_seq_length, d_model) for _ in range(n_layers)]
        self.dec_embed = tf.keras.layers.Embedding(vocab_size, d_model)

        # Decoder Layers. #
        self.dec_layers = [DecoderLayer(
            d_model, n_heads, d_ffwd, rate1, rate2) for _ in range(n_layers)]
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
    
    def call(self, x, training=True, logits=True):
        seq_length = tf.shape(x)[1]
        input_mask = tf.linalg.band_part(
            tf.ones([seq_length, seq_length]), -1, 0)
        input_mask = 1.0 - input_mask
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            layer_output = self.dec_layers[m](
                layer_input, x_pos_embed, 
                training=training, mask=input_mask)
            layer_input  = layer_output

        # Return the vocab logits. #
        if not logits:
            return layer_output
        else:
            # Extract the embedding matrix. #
            x_vocab = tf.range(self.vocab_size)
            w_embed = self.dec_embed(x_vocab)

            dec_logits = tf.matmul(
                layer_output, w_embed, transpose_b=True)
            return dec_logits

class AFTDecoder(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, d_model, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(AFTDecoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.vocab_size = vocab_size
        
        # Output projection. #
        self.aft_model = Decoder(
            n_layers, d_model, 
            n_heads, d_ffwd, vocab_size, 
            max_seq_length, rate1=rate1, rate2=rate2)
    
    def call(self, x, training=True):
        dec_logits = self.aft_model(
            x, training=training, logits=True)
        return dec_logits
    
    def compute_xent_loss(self, x_input, x_label):
        seq_length = x_input.shape[1]
        dec_states = self.aft_model(
            x_input, training=True, logits=False)
        
        # Extract the embedding matrix. #
        x_vocab = tf.range(self.vocab_size)
        w_embed = self.aft_model.dec_embed(x_vocab)
        
        # Manually compute the cross entropy loss. #
        xent_losses = []
        for t_step in range(seq_length):
            dec_logit = tf.matmul(
                dec_states[:, t_step, :], w_embed, transpose_b=True)
            
            tmp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=x_label[:, t_step], logits=dec_logit)
            xent_losses.append(tf.expand_dims(tmp_loss, axis=1))
        
        xent_losses = tf.concat(xent_losses, axis=1)
        return xent_losses

    def infer(self, x):
        """
        To be depreciated. Use self.gen_text() instead.
        """
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        for step in range(self.seq_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_logits = self.call(tmp_inputs, training=False)
            
            tmp_logit = tmp_logits[:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
    
    def gen_text(
        self, x, gen_len=None, sample=True):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        if gen_len is None:
            gen_len = self.seq_len
        
        for step in range(gen_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_logits = self.call(
                tmp_inputs[:, -self.seq_len:], training=False)
            
            tmp_logit = tmp_logits[:, -1, :]
            if sample:
                tmp_probs  = tf.nn.softmax(
                    tmp_logit, axis=1).numpy()[0, :]
                tmp_sample = np.random.choice(
                    self.vocab_size, p=tmp_probs)
                tmp_sample = tf.expand_dims(tf.constant(
                    tmp_sample, dtype=tf.int32), axis=0)
            else:
                tmp_sample = tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32)
            
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], lambda: tmp_sample)
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)

