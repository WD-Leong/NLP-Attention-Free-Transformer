# Import the libraries. #
import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras as tf_gpt

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            raw_losses = model.compute_xent_loss(tmp_encode, tmp_output)
            tmp_losses = tf.reduce_sum(tf.reduce_sum(raw_losses, axis=1))
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_losses

# Model Parameters. #
prob_keep  = 0.9
batch_size = 256
sub_batch  = 256
num_heads  = 4
num_layers = 8
seq_length = 31

gradient_clip = 1.00
maximum_iter  = 10000
restore_flag  = False
save_step     = 500
warmup_steps  = 5000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 100

model_ckpt_dir  = "../TF_Models/dialogue_subword_gpt_8_layers"
train_loss_file = "train_loss_dialogue_subword_gpt_8_layers.csv"

# Load the data. #
tmp_pkl_file = "../Data/movie_dialogs/movie_dialogues_sw.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

# Filter the dataset. #
filtered_data = []
for tmp_data in data_tuple:
    tot_len = len(tmp_data[0]) + len(tmp_data[1])
    if tot_len > 1 and tot_len <= (seq_length-1):
        filtered_data.append(tmp_data)
del tmp_data, data_tuple

data_tuple = filtered_data
vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size))
del filtered_data

num_data  = len(data_tuple)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]
print("Total of", str(len(data_tuple)), "rows loaded.")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the GPT Keras Model.")
start_time = time.time()

gpt_model = tf_gpt.GPTDecoder(
    num_layers, num_heads, 
    hidden_size, ffwd_size, vocab_size, 
    seq_length, rate1=0.0, rate2=1.0-prob_keep)
gpt_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("GPT Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Train the GPT model. #
tmp_out_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.005
    anneal_pow = int(n_iter / anneal_step)
    learning_rate = max(np.power(
        anneal_rate, anneal_pow)*initial_lr, 2.5e-5)

print("-" * 50)
print("Training the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_pow = int(n_iter / anneal_step)
            learning_rate = max(np.power(
                anneal_rate, anneal_pow)*initial_lr, 2.5e-5)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_idx = data_tuple[tmp_index][0]
        tmp_o_idx = data_tuple[tmp_index][1]
        tmp_p_idx = tmp_i_idx + [SOS_token] + tmp_o_idx
        
        n_input = len(tmp_p_idx)
        tmp_out_seq[n_index, :n_input] = tmp_p_idx
        tmp_out_seq[n_index, n_input]  = EOS_token
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    tmp_loss = sub_batch_train_step(
        gpt_model, sub_batch, tmp_input, tmp_output, 
        gpt_optimizer, learning_rate=learning_rate)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        avg_ppl  = np.log2(avg_loss)
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        tmp_test_in[:, :] = PAD_token
        sample_id = np.random.choice(num_data, size=1)
        tmp_data  = data_tuple[sample_id[0]]
        tmp_i_idx = tmp_data[0]
        tmp_i_tok = bpe.bp_decode(tmp_i_idx, idx_2_subword)
        tmp_o_tok = bpe.bp_decode(tmp_data[1], idx_2_subword)
        n_tokens  = len(tmp_i_idx) + 1
        
        tmp_in_phrase  = " ".join(
            tmp_i_tok).replace("<", "").replace(">", "")
        tmp_out_phrase = " ".join(
            tmp_o_tok).replace("<", "").replace(">", "")
        tmp_test_in[0, :n_tokens] = tmp_i_idx + [SOS_token]
        
        gen_tokens = gpt_model.infer(
            tmp_test_in[:, :n_tokens]).numpy()[0]
        gen_phrase = bpe.bp_decode(
            gen_tokens, idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        gen_output = bpe.bp_decode(
            gen_tokens[(n_tokens-1):], idx_2_subword)
        gen_output = " ".join(
            gen_output).replace("<", "").replace(">", "")
        
        print("Iteration", str(n_iter) + ".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        print("Average Perplexity:", str(avg_ppl) + ".")
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Generated Response:")
        print(gen_output)
        print("Actual Response:")
        print(tmp_out_phrase)
        del n_tokens, sample_id
        
        train_loss_list.append((n_iter, avg_loss, avg_ppl))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_loss_cols = ["n_iter", "xent_loss", "perplexity"]
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=tmp_loss_cols)
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

