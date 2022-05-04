import os
import json
import pandas as pd
import numpy as np
from pprint import pprint

vocab_file = 'data/token_vec_300.bin'

def get_embed(file):
    word2ix = {}   # 词 --->id
    row = 1
    word2embed = {}   # 词---->嵌入

    word2ix['PAD&UNK'] = 0
    word2embed['PAD&UNK'] = [float(0)] * 300

    for line in open(file, 'r').readlines():
        data = line.strip().split()
        word = data[0]
        embed = data[1:]
        embed = [float(num) for num in embed]
        word2embed[word] = embed
        word2ix[word] = row
        row += 1

    ix2word = {ix:word for word, ix in word2ix.items()}
    id2embed = {}
    for ix in range(len(word2ix)):
        id2embed[ix] = word2embed[ix2word[ix]]

    embed = np.array([id2embed[ix] for ix in range(len(word2ix))])
    return embed, word2ix, ix2word

'''对文本长度进行截取与填充'''
def padding(text, maxlen=20):
    pad_text = []
    for sentence in text:
        pad_sentence = np.zeros(maxlen).astype('int64')
        cnt = 0
        for index in sentence:
            pad_sentence[cnt] =index
            cnt += 1
            if cnt == maxlen:
                break
        pad_text.append(pad_sentence.tolist())
    return pad_text

'''批量返回文本对的索引'''
def char_index(text_a, text_b, file):
    embed, char2ix, ix2char = get_embed(file)
    a_list, b_list = [], []

    for a_sentence, b_sentence in zip(text_a, text_b):
        a, b = [], []
        for char in str(a_sentence).lower():
            if char in char2ix.keys():
                a.append(char2ix[char])
            else:
                a.append(0)

        for char in str(b_sentence).lower():
            if char in char2ix.keys():
                b.append(char2ix[char])
            else:
                b.append(0)
        a_list.append(a)
        b_list.append(b)
    a_list = padding(a_list)
    b_list = padding(b_list)
    return a_list, b_list

'''给定数据路径，返回文本对的索引以及标签'''
def load_char_data(filename, file):
    df = pd.read_csv(filename, encoding='utf-8', sep='\t')
    text_a = df['text_a'].values
    text_b = df['text_b'].values
    label = df['label'].values
    a_index, b_index = char_index(text_a, text_b, file)
    return np.array(a_index), np.array(b_index), np.array(label)


