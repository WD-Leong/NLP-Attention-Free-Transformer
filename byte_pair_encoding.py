# -*- coding: utf-8 -*-
"""
Created on Wed May 20 00:21:52 2020

@author: admin
"""

import collections
from collections import Counter
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split(" ")
        if len(symbols) > 1:
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

def learn_subword_vocab(
    word_vocab, n_iter, vocab_size=8000, verbose=False):
    for m in range(n_iter):
        pairs = get_stats(word_vocab)
        if len(pairs) >= 1:
            most_freq  = max(pairs, key=pairs.get)
            word_vocab = merge_vocab(most_freq, word_vocab)
        else:
            print("No more sub-words to iterate.")
            break
        
        if verbose:
            if (m+1) % 100 == 0:
                print(str(m+1), "merges out of", 
                      str(n_iter), "complete.")
    
    # Get all the subwords from pairs dictionary. #
    subword_vocab = collections.defaultdict(int)
    for word, freq in pairs.items():
        if freq == 0:
            continue
        subword_vocab[word[0]] += freq
        subword_vocab[word[1]] += freq
    del word, freq
    
    # Extend the dictionary to include full words. #
    for word, freq in word_vocab.items():
        symbols = word.split(" ")
        if len(symbols) == 1:
            subword_vocab[word] += freq
    
    special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    subword_vocab_list = sorted(
        subword_vocab, key=subword_vocab.get, reverse=True)[:vocab_size]
    
    if "<" not in subword_vocab_list:
        special_tokens += ["<"]
    if ">" not in subword_vocab_list:
        special_tokens += [">"]
    subword_vocab_list = special_tokens + subword_vocab_list
    
    idx2subword = dict([(x, subword_vocab_list[x]) for x \
                        in range(len(subword_vocab_list))])
    subword2idx = dict([(subword_vocab_list[x], x) for x \
                        in range(len(subword_vocab_list))])
    return subword_vocab_list, idx2subword, subword2idx

def learn_word_vocab(corpus):
    w_counter = Counter()
    
    for tmp_text in corpus:
        tmp_tokens = word_tokenizer(tmp_text.strip().lower())
        w_counter.update(tmp_tokens)
    
    word_counts = []
    for word, count in w_counter.items():
        tmp_word = "<" + word + ">"
        tmp_word = "".join([x+" " for x in tmp_word]).strip()
        word_counts.append((tmp_word, count))
    return dict(word_counts)

def subword_tokenize(word, bp_vocab, subword2idx):
    sow = "<"
    eow = ">"
    word_pad  = sow + word.strip() + eow
    sw_tokens = []
    
    st_idx = 0
    en_idx = len(word_pad)
    while st_idx < len(word_pad):
        subword = word_pad[st_idx:en_idx]
        if subword in bp_vocab:
            sw_tokens.append(subword)
            st_idx = en_idx
            en_idx = len(word_pad)
        elif len(subword) == 1:
            sw_tokens.append("<UNK>")
            st_idx = en_idx
            en_idx = len(word_pad)
        else:
            en_idx -= 1
    
    id_subwords = [subword2idx[x] for x in sw_tokens]
    return id_subwords

def bp_encode(text_in, bp_vocab, subword2idx):
    sw_token_list = []
    for tmp_token in \
        word_tokenizer(text_in.lower().strip()):
        sw_token_list.extend(subword_tokenize(
            tmp_token, bp_vocab, subword2idx))
    return sw_token_list

def bp_decode(indices_in, idx2subword):
    sw_idx_list = [idx2subword[x] for x in indices_in]
    words_list  = []
    
    curr_sw = ""
    for n_sw in range(len(sw_idx_list)):
        tmp_sw = sw_idx_list[n_sw]
        if tmp_sw.find("<") != -1 \
            and tmp_sw.find(">") != -1:
            tmp_word = tmp_sw
            curr_sw = ""
            words_list.append(tmp_word)
        elif tmp_sw.find(">") != -1 \
            and tmp_sw.find("<") == -1:
            curr_sw += tmp_sw
            tmp_word = curr_sw
            curr_sw = ""
            words_list.append(tmp_word)
        elif tmp_sw.find(">") == -1 \
            and tmp_sw.find("<") != -1:
            curr_sw += tmp_sw
        else:
            curr_sw += tmp_sw
    return words_list
