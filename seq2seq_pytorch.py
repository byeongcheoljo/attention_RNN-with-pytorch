import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import time
from torch.utils.data import Dataset, DataLoader

print(torch.__version__)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()

    # adding a start and an end token to the sentence, especially sequence to sequence model know when to start and stop predicting.
    #eg: "he is a boy ."  => "<start> he is a boy . <end>"
    w = '<start> ' + w + ' <end>'
    return w

def maxSequenceLength(tensor):
    return max(len(x) for x in tensor)

## padding값을 넣어주는 함수
def padSequence(x, max_length):
    padded = np.zeros(max_length, dtype=np.int64)
    if len(x) > max_length: padded[:] = x[:max_length]
    else: padded[:len(x)] = x
    return padded

file_name = "/home/byeongcheol/python3_deep_env/pytorch_test/data/eng-fra.txt"
eng_list = []
fra_list = []
for ln in open(file_name, 'rt',encoding="utf-8"):
    eng, fra = ln.strip().split("\t")
    fra = fra.replace("\u202f", ' ')
    eng_list.append(preprocess_sentence(eng))
    fra_list.append(preprocess_sentence(fra))


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2index = {}
        self.index2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(" "))

        self.vocab = sorted(self.vocab)
        self.word2index['<pad>'] = 0

        for index, word in enumerate(self.vocab):
            self.word2index[word] = index + 1

        for word, index in self.word2index.items():
            self.index2word[index] = word


input_lang = LanguageIndex(fra_list)
target_lang = LanguageIndex(eng_list)
input_tensor = [[input_lang.word2index[s] for s in fra.split(' ')]  for fra in fra_list]
target_tensor = [[target_lang.word2index[word] for word in es.split(" ")] for es in eng_list]
print(input_tensor[0:10])

## encoder, decoder에 문장 최대 길이를 구한다.
input_tensor_max_length, target_tensor_max_length = maxSequenceLength(input_tensor), maxSequenceLength(target_tensor)
print(input_tensor_max_length, target_tensor_max_length)
##encoder, decoder에 padding 값 추가
input = ([padSequence(x, input_tensor_max_length) for x in input_tensor])
target = ([padSequence(y, target_tensor_max_length) for y in target_tensor])

print(len(input[0]))
# print(target[135841 : 135842])

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input, target, test_size=0.2)
print(len(input_tensor_train), len(input_tensor_val), len(target_tensor_train), len(target_tensor_val))


class MyData(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.target = Y
        self.length = [np.sum(1- np.equal(x,0)) for x in X]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len

    def __len__(self):
        return len(self.data)

batch_size = 64
embedding_dim = 10
hidden_size = 1024
vocab_inp_size = len(input_lang.word2index)
vocab_tar_size = len(target_lang.word2index)


train_dataset = MyData(input_tensor_train, target_tensor_train)
val_dataset = MyData(input_tensor_val, target_tensor_val)


dataset = DataLoader(train_dataset, batch_size = batch_size,
                     drop_last=False,
                     shuffle=False)



class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, batch_first = True)

    def forward(self, x, lens):
        # x: batch_size, max_length

        # x: batch_size, max_length, embedding_dim
        x = self.embedding(x)
        x = x.view(batch_size, 65, embedding_dim)
        print(x.shape,"XXXX")
        # x transformed = max_len X batch_size X embedding_dim
        # x = x.permute(1,0,2)
        x = pack_padded_sequence(x, lens, batch_first=True) # unpad

        self.hidden = self.initialize_hidden_state()

        # output: max_length, batch_size, hidden_size
        # self.hidden: 1, batch_size, hidden_size
        output, self.hidden = self.gru(x, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep

        # pad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output, batch_first = True)

        return output, self.hidden

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_size, self.hidden_size))

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

encoder = Encoder(vocab_inp_size, embedding_dim, hidden_size, batch_size)
###test model output
it = iter(dataset)
x, y, x_len = next(it)
print(x.shape)
# sort the batch first to be able to use with pac_pack_sequence
xs, ys, lens = sort_batch(x, y, x_len)
print(xs, ys, lens)
enc_output, enc_hidden = encoder(xs, lens)
print(enc_output.size()) # max_length, batch_size, hidden_size
print(enc_hidden.size())

