#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import pickle

PAD_CHAR = '\1'
UNK_CHAR = '\0' 
BATCH_SIZE = 512 
HIDDEN_DIM = 128
N_RNN_LAYERS = 2
N_EPOCHS = 10
LEARNING_RATE = 1e-2


def apply_vocab(data, char_to_index):
    for itr in range(len(data)):
        data[itr] = [char_to_index.get(character, char_to_index[UNK_CHAR]) for character in data[itr]]

# set up one-hot encodings
def get_features(lines, char_to_index, sequence_len, batch_size=BATCH_SIZE):
    features = np.zeros((batch_size, sequence_len, len(char_to_index.items())), dtype=np.float32)
    lines = np.array(lines)

    # print(f"lines shape for epoch: {lines.shape}")
    # print(f"features shape for epoch: {features.shape}")
    # print(f"batch_size: {batch_size}")
    # print(f"sequence_len: {sequence_len}")
    for batch_itr in range(batch_size):
        for itr in range(sequence_len):
            features[batch_itr, itr, lines[batch_itr][itr]] = 1

    return features

# set up one-hot encodings
def get_features_single(lines, char_to_index, sequence_len):
    features = np.zeros((len(lines), sequence_len, len(char_to_index.items())), dtype=np.float32)
    # print('==========')
    # print(features.shape)
    # print(lines)
    # print(len(lines))

    for batch_itr in range(len(lines)):
        for itr in range(sequence_len):
            features[batch_itr, itr, lines[itr]] = 1
    return features

# we referenced this basic tutorial when setting up our rnn
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class RNN_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        # print(f"shape {tuple(x.shape)[1]}")
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        # hidden = hidden + (0.1**0.5)*torch.randn(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    m: RNN_Model

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        data = []
        f = open('data/corpus.txt', "r", encoding='utf-8')
        for line in f:
            data.append(line)

        # cut out excess data so the count of datas can be evenly divided
        # by the batch number
        limit = BATCH_SIZE * (len(data) // BATCH_SIZE)
        org_length = len(data)
        for _ in range(limit, org_length):
            data.pop()


        # print("Run stats:")
        # print(f"N: {len(data)}")

        char_to_index = {PAD_CHAR: 0, UNK_CHAR: 1}
        longest_len = 0

        sum_len = 0

        chars = set()

        for line in data:
            sum_len += len(line)
            if len(line) > longest_len:
                longest_len = len(line)

            for char_itr in range(len(line)):
                character = line[char_itr]
                if character not in chars:
                    char_to_index[character] = len(char_to_index.items())
                    chars.add(character)

        longest_len = sum_len // len(data)

        for itr in range(len(data)):
            if len(data[itr]) > longest_len:
                data[itr] = data[itr][:longest_len]
            else:
                pad_len = longest_len - len(data[itr])
                # does it matter whether we pad in the front or at the end?
                data[itr] = data[itr] + (PAD_CHAR * pad_len)

        data_X = [line[:-1] for line in data]
        data_Y = [line[1:] for line in data]

        for i in range(len(data_X)):
            assert(len(data_X[i]) == longest_len - 1)

        apply_vocab(data_X, char_to_index)
        apply_vocab(data_Y, char_to_index)

        print(f"data_X: {len(data_X)}")
        print(f"data_Y: {len(data_Y)}")
        print(f"Number of unique characters in dataset (+ UNK, PAD): {len(char_to_index)}")
        return (data_X, data_Y, char_to_index, longest_len - 1)

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, X, Y, char_to_index, work_dir, longest_len):
        cuda_available = torch.cuda.is_available()
        print(f"Cuda check = {cuda_available}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.m = RNN_Model(input_size=len(char_to_index), output_size=len(char_to_index), hidden_dim=HIDDEN_DIM, n_layers=N_RNN_LAYERS)
        optimizer = torch.optim.Adam(self.m.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, N_EPOCHS + 1):
            print(f"Epoch: {epoch}")
            batch_X = [X[i:i+BATCH_SIZE] for i in range(0, len(X), BATCH_SIZE)]
            batch_Y = [Y[i:i+BATCH_SIZE] for i in range(0, len(Y), BATCH_SIZE)]
            for itr in range(len(batch_X)):
                one_hot_matrix = get_features(batch_X[itr], char_to_index, longest_len)
                input_vec = torch.from_numpy(one_hot_matrix)
                output_vec = torch.Tensor(batch_Y[itr])
                self.train_batch(optimizer, criterion, device, input_vec, output_vec)

        return self.m


    def train_batch(self, optimizer, criterion, device, X, Y):
        optimizer.zero_grad()
        X.to(device)
        Y.to(device)
        output, hidden = self.m(X)

        # print('-----------------')
        # print(output.shape)
        # print(X.shape)
        # print(Y.shape)
        # print(Y.view(-1).long().shape)

        loss = criterion(output, Y.view(-1).long())
        loss.backward()
        optimizer.step()
        # print(f"Loss: {loss.item()}")


    def run_pred(self, data, char_to_index):
        # your code here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        index_to_char = {v: k for k, v in char_to_index.items()}
        apply_vocab(data, char_to_index)

        preds = []
        for line in data:
            # this model just predicts a random character each time
            line_encoding = get_features_single(line, char_to_index, len(line))
            line_encoding = torch.from_numpy(line_encoding)
            line_encoding.to(device)

            out, hidden, = self.m(line_encoding)

            prob = nn.functional.softmax(out[-1], dim=0).data
            char_ind = torch.max(prob, dim=0)[1].item()

            bad_pred_set = {' ', PAD_CHAR, UNK_CHAR, '\n'}
            top7_letters, top7_indices = torch.topk(prob, 7, dim=0)
            top3_letters = []
            top3_indices = []
            itr = 0
            while len(top3_letters) < 3:
                pred = top7_indices.numpy()[itr]
                if index_to_char[pred] not in bad_pred_set:
                    top3_letters.append(index_to_char[pred])
                    top3_indices.append(pred)
                itr += 1


            top_guesses = '' + top3_letters[0] + top3_letters[1] + top3_letters[2]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir, m, char_to_index):
        torch.save(self.m, work_dir + '/trained_model.model')
        pickle.dump(char_to_index, open('./work/char_to_index.pickle', 'wb'))

    @classmethod
    def load(cls, work_dir):
        model = torch.load(work_dir + '/trained_model.model')
        char_to_index = pickle.load(open('./work/char_to_index.pickle', 'rb'))
        return model, char_to_index


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        X, Y, char_to_index, longest_len = MyModel.load_training_data()
        print('longest len = ', longest_len)
        print('Training')
        m = model.run_train(X, Y, char_to_index, args.work_dir, longest_len)
        print('Saving model')
        model.save(args.work_dir, m, char_to_index)
    elif args.mode == 'test':
        print('Loading model')
        m, char_to_index = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        model = MyModel()
        model.m = m
        pred = model.run_pred(test_data, char_to_index)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
