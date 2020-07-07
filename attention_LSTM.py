import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
# 1 is good, 0 is bad.
labels = [1, 1, 1, 0, 0, 0]

word_list = list(set(" ".join(sentences).split()))
word2index = {word: index for index, word in enumerate(word_list)}

vocab_size = len(word2index)
input_dim = 10
hidden_size = 15
num_classes = len(list(set(labels)))

#Convert word for sentence to integer.
input_sentence = []
for sent in sentences:
    input_sentence.append(np.asarray([word2index[word] for word in sent.split()]))

target = []
for label in labels:
    target.append(label)

input = torch.LongTensor(input_sentence)
output = torch.LongTensor(target)

class Model(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = input_dim)
        self.lstm_layer = nn.LSTM(input_dim, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes) # if bidirectional == True, (hidden_size * 2, output)

    def attention(self, lstm_output, lstm_hidden_state):
        hidden = lstm_hidden_state.view(-1, hidden_size, 1)#[batch_size, hidden_size * num_directions, layer]
        attention_weight = torch.bmm(lstm_output, hidden).squeeze(2) # [batch, sequence_length]
        softmax_atten_weight = F.softmax(attention_weight)
         # [batch_size, hidden_size * num_directions(=2), sequence_length] * [batch_size, sequence_length, 1] = [batch_size, hidden_size * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1,2), softmax_atten_weight.unsqueeze(2)).squeeze(2)
        # [batch_size, hidden_size * num_bidirectional]
        return context, softmax_atten_weight

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(-1, 3, input_dim) # [batch_size, sequence_length, input_dim]
        # hidden_0 = torch.zeros(1, 6, hidden_size) # [numlayer * num_bidirectional, batch_size, hidden_size]
        # cell_0 = torch.zeros(1,6,hidden_size) # [num_layer * num_bidirectional, batch_size, hidden_size]
        # out, (hidden, cell) = self.lstm_layer(x, (hidden_0, cell_0))
        out, (hidden, cell) = self.lstm_layer(x)
        attention_output, attention = self.attention(out, hidden)
        return self.fc(attention_output), attention

model = Model(vocab_size, input_dim, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training
for epoch in range(50):
    optimizer.zero_grad()
    pred, attention = model(input)
    loss = criterion(pred, output)
    if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict of train data
predict, _ = model(input)
predict = predict.data.max(1, keepdim=True)[1]
result = (predict.tolist())
for i in result:
    if 1 in i:
        print("Good")
    else:
        print("Bad")

####test
test_text = 'she loves you'
tests = [np.asarray([word2index[n] for n in test_text.split()])]
test_batch = torch.LongTensor(tests)
test = (test_batch)
predict, _ = model(test)

result_predict = ((predict.data.max(1, keepdim=True)[1]))

if result_predict[0][0] == 0:
    print("Bad :(")

else:
    print("Good :)")
