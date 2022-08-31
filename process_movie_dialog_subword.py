import re
import time
import collections
import pickle as pkl
from collections import Counter
import byte_pair_encoding as bpe
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split(" ")
        for m in range(len(symbols)-1):
            pairs[symbols[m], symbols[m+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = " ".join(pair)
    for word in v_in:
        w_out = word.replace(bigram, "".join(pair))
        v_out[w_out] = v_in[word]
    return v_out

def learn_subword_vocab(word_vocab, n_iter):
    for m in range(n_iter):
        pairs = get_stats(word_vocab)
        most_freq  = max(pairs, key=pairs.get)
        word_vocab = merge_vocab(most_freq, word_vocab)
    
    subword_vocab = collections.defaultdict(int)
    for word, freq in pairs.items():
        if freq == 0:
            continue
        subword_vocab[word[0]] += freq
        subword_vocab[word[1]] += freq
    return subword_vocab, pairs

print("Loading the data.")
tmp_path = "../../Data/movie_dialogs/"
start_tm = time.time()

tmp_line_file = tmp_path + "movie_lines.txt"
with open(
    tmp_line_file, "r", 
    encoding='utf-8', errors='ignore') as tmp_file:
    tmp_lines = tmp_file.readlines()

tmp_conv_file = tmp_path + "movie_conversations.txt"
with open(
    tmp_conv_file, "r", 
    encoding='utf-8', errors='ignore') as tmp_file:
    tmp_convs = tmp_file.readlines()

id2line = {}
for tmp_line in tmp_lines:
    tmp_split = str(tmp_line).split(" +++$+++ ")
    if len(tmp_split) == 5:
        id2line[tmp_split[0]] = tmp_split[4]

convs = []
for tmp_conv in tmp_convs[:-1]:
    tmp_split = str(tmp_conv).replace("\\n", "").split(
        " +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    tmp_split = tmp_split.replace("]", "")
    
    tmp_ids = [str(x.encode("utf-8")).replace(
        "b'", "").replace("'", "") for x in tmp_split.split(",")]
    convs.append(tmp_ids)

q_len = 10
a_len = 10

w_counter  = Counter()
tmp_corpus = []
tmp_data_tuple = []
for conv in convs:
    for i in range(len(conv)-1):
        tmp_qns = id2line[conv[i]].lower().replace(
            "\\u", " ").replace("\\i", " ").replace("\n", " ").replace("\t", " ")
        #tmp_qns = re.sub(r"[^\w\s]", " ", tmp_qns)
        tmp_qns = [x for x in word_tokenizer(tmp_qns) if x != ""]

        tmp_ans = id2line[conv[i+1]].lower().replace(
            "\\u", " ").replace("\\i", " ").replace("\n", " ").replace("\t", " ")
        #tmp_ans = re.sub(r"[^\w\s]", " ", tmp_ans)
        tmp_ans = [x for x in word_tokenizer(tmp_ans) if x != ""]
        
        if len(tmp_qns) == 0 or len(tmp_ans) == 0:
            continue
        elif len(tmp_qns) <= q_len and len(tmp_ans) <= a_len:
            w_counter.update(tmp_qns)
            w_counter.update(tmp_ans)
            tmp_data_tuple.append((" ".join(tmp_qns), " ".join(tmp_ans)))

elapsed_tm = (time.time() - start_tm) / 60
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Fit the subword vocabulary. #
print("Fitting subword vocabulary.")
start_tm = time.time()

word_counts = []
for word, count in w_counter.items():
    tmp_word = "<" + word + ">"
    tmp_word = "".join([x+" " for x in tmp_word]).strip()
    word_counts.append((tmp_word, count))
word_counts = dict(word_counts)

n_iters = 1000
vocab_size = 8000
tuple_out  = bpe.learn_subword_vocab(
    word_counts, n_iters, vocab_size=vocab_size)

subword_vocab_list = tuple_out[0]
idx2subword = tuple_out[1]
subword2idx = tuple_out[2]

elapsed_tm = (time.time() - start_tm) / 60
print("Total Sub-word Vocabulary size:", 
      str(len(subword_vocab_list)), "sub-word tokens")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Encode the corpus to subword tokens. #
print("Encoding the corpus to subwords.")
start_tm = time.time()

new_tuple = []
for tmp_qns, tmp_ans, in tmp_data_tuple:
    tmp_qns_sw = bpe.bp_encode(
        tmp_qns, subword_vocab_list, subword2idx)
    tmp_ans_sw = bpe.bp_encode(
        tmp_ans, subword_vocab_list, subword2idx)
    new_tuple.append((tmp_qns_sw, tmp_ans_sw))

elapsed_tm = (time.time() - start_tm) / 60
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Save the data. #
print("Saving the file.")
tmp_pkl_file = tmp_path + "movie_dialogues_subword.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(new_tuple, tmp_file_save)
    pkl.dump(subword_vocab_list, tmp_file_save)
    pkl.dump(idx2subword, tmp_file_save)
    pkl.dump(subword2idx, tmp_file_save)