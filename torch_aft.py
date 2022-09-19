import torch
import torch.nn.functional as F

# Attention-Free Layer. #
class AttnFreeLayer(torch.nn.Module):
    def __init__(self, d_model):
        super(AttnFreeLayer, self).__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.d_model = d_model
        
        self.wq = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(
            d_model, d_model, bias=False)
    
    def forward(self, v, k, q):
        #q_input = self.sigmoid(self.wq(q))
        #k_input = torch.exp(self.wk(k))
        q_input = F.elu(self.wq(q)) + 1.0
        k_input = F.elu(self.wk(k)) + 1.0
        v_input = self.wv(v)

        # Prefix sums for causality. #
        kv_input  = torch.mul(k_input, v_input)
        k_prefix  = torch.cumsum(k_input, dim=1)
        kv_prefix = torch.cumsum(kv_input, dim=1)
        
        kv_softmax = torch.div(
            kv_input + kv_prefix, k_prefix)
        attn_outputs = torch.mul(q_input, kv_softmax)
        return attn_outputs

class FFWNetwork(torch.nn.Module):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = torch.nn.Linear(d_model, d_ffwd)
        self.ffwd_2 = torch.nn.Linear(d_ffwd, d_model)
    
    def forward(self, x):
        return self.ffwd_2(
            torch.square(F.relu(self.ffwd_1(x))))

# GPT Decoder Layer. #
class DecoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, d_ffwd, rate1=0.1, rate2=0.1):
        super(DecoderLayer, self).__init__()
        self.rate1 = rate1
        self.rate2 = rate2
        self.attn_self = AttnFreeLayer(d_model)
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        
        self.lnorm_1 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_2 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        
        self.dropout_1 = torch.nn.Dropout(rate1)
        self.dropout_2 = torch.nn.Dropout(rate2)
    
    def forward(
        self, x_enc, x_pos, training=True):
        x_embed = x_enc + x_pos
        attn_outputs = self.lnorm_1(
            self.attn_self(x_embed, x_embed, x_embed))
        
        # Apply normalization before residual connection. #
        attn_outputs = torch.add(x_embed, attn_outputs)
        if training:
            attn_outputs = self.dropout_1(attn_outputs)
        
        ffwd_outputs = self.lnorm_2(
            self.ffwd_self(attn_outputs))
        ffwd_outputs = torch.add(
            attn_outputs, ffwd_outputs)
        if training:
            ffwd_outputs = self.dropout_2(ffwd_outputs)
        return ffwd_outputs

class Decoder(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, d_ffwd, vocab_size, 
        max_seq_length, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = torch.rsqrt(
            torch.as_tensor(d_model, dtype=torch.float32))
        self.vocab_size = vocab_size

        # Embedding Layers. #
        self.pos_embed = torch.nn.ModuleList()
        self.dec_embed = torch.nn.Embedding(vocab_size, d_model)
        
        # Decoder Layers. #
        self.dec_layers = torch.nn.ModuleList()
        for m in range(n_layers):
            self.pos_embed.append(
                torch.nn.Embedding(max_seq_length, d_model))
            self.dec_layers.append(
                DecoderLayer(d_model, d_ffwd, rate1, rate2))
        self.emb_dropout = torch.nn.Dropout(rate1)
    
    def forward(self, x, training=True):
        seq_length = list(torch._shape_as_tensor(x))[1]
        
        x_pos_index = torch.unsqueeze(torch.arange(
            0, seq_length, dtype=torch.long), 0)
        if torch.cuda.is_available():
            x_pos_index = x_pos_index.to("cuda")

        x_tok_embed = self.dec_embed(x)
        x_tok_embed = x_tok_embed * self.d_rsqrt
        if training:
            x_tok_embed = self.emb_dropout(x_tok_embed)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = x_pos_embed * self.d_rsqrt
            if training:
                x_pos_embed = self.emb_dropout(x_pos_embed)
            
            layer_output = self.dec_layers[m](
                layer_input, x_pos_embed, training=training)
            layer_input  = layer_output
        return layer_output

class AFTDecoder(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, d_ffwd, vocab_size, 
        max_seq_length, rate1=0.1, rate2=0.1):
        super(AFTDecoder, self).__init__()
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        
        self.vocab_size = vocab_size
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(
            reduction="none")
        
        # Output projection. #
        self.gpt_model = Decoder(
            n_layers, d_model, d_ffwd, vocab_size, 
            max_seq_length, rate1=rate1, rate2=rate2)
        self.p_decoder = torch.nn.Linear(
            d_model, vocab_size, bias=False)
    
    def forward(self, x, training=True):
        dec_outputs = self.gpt_model(
            x, training=training)
        return dec_outputs
    
    def compute_ce_loss(
        self, x_input, x_output, seg_len=None):
        if seg_len is None:
            seg_len = self.seq_len
        
        if self.seq_len <= seg_len:
            n_segments = 1
        elif self.seq_len % seg_len == 0:
            n_segments = int(self.seq_len / seg_len)
        else:
            n_segments = int(self.seq_len / seg_len) + 1

        seq_ce_loss = 0.0
        tmp_dec_out = self.forward(
            x_input, training=True)
        for n_segment in range(n_segments):
            l_st = n_segment * seg_len
            if n_segment != (n_segments-1):
                l_en = (n_segment+1) * seg_len
            else:
                l_en = self.seq_len
            
            tmp_labels  = x_output[:, l_st:l_en]
            tmp_segment = tmp_dec_out[:, l_st:l_en, :]
            tmp_logits  = self.p_decoder(tmp_segment)
            seg_ce_loss = torch.sum(torch.sum(self.ce_loss_fn(
                torch.transpose(tmp_logits, 1, 2), tmp_labels), 1))
            seq_ce_loss += seg_ce_loss
        return seq_ce_loss
    
    def infer(self, x, k=1):
        input_len = list(torch._shape_as_tensor(x))[1]
        infer_ids = [torch.unsqueeze(x[:, 0], 1)]
        
        for step in range(self.seq_len):
            tmp_inputs = torch.cat(infer_ids, dim=1)
            with torch.no_grad():
                tmp_outputs = self.forward(
                    tmp_inputs, training=False)
                tmp_logits  = self.p_decoder(tmp_outputs)
                
                if step < (input_len-1):
                    tmp_argmax = x[:, step+1]
                    infer_ids.append(torch.unsqueeze(tmp_argmax, 1))
                else:
                    if k == 1:
                        tmp_argmax =  torch.argmax(
                            tmp_logits[:, -1, :], dim=1)
                        infer_ids.append(torch.unsqueeze(tmp_argmax, 1))
                    else:
                        tmp_logit = tmp_logits[:, -1, :]
                        tmp_prob  = F.softmax(tmp_logit, dim=1)

                        tmp_top_k  = torch.topk(tmp_prob, k=k)
                        tmp_sample = torch.multinomial(
                            tmp_top_k.values, 1)
                        tmp_index  = torch.gather(
                            tmp_top_k.indices, 1, tmp_sample)
                        infer_ids.append(tmp_index)
        return torch.cat(infer_ids, dim=1)
