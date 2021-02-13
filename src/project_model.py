#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

PAD_CHAR = ' '
BATCH_SIZE = 128
HIDDEN_DIM = 256
N_RNN_LAYERS = 2
N_EPOCHS = 20
LEARNING_RATE = 1e-3


def apply_vocab(data, char_to_index):
    for itr in range(len(data)):
        data[itr] = [char_to_index[character] for character in data[itr]]

# set up one-hot encodings
def get_features(lines, char_to_index, sequence_len):
    features = np.zeros((BATCH_SIZE, sequence_len, len(char_to_index.items())), dtype=np.float32)

    for batch_itr in range(BATCH_SIZE):
        for itr in range(sequence_len):
            features[batch_itr, itr, lines[batch_itr][itr]] = 1
    return features

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
        return hidden

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        data = []
        f = open('data/dialogue_short.txt', "r", encoding='utf-8')
        for line in f:
            data.append(line)

        char_to_index = {PAD_CHAR: 0}
        longest_len = 0
        chars = set()

        for line in data:
            if len(line) > longest_len:
                longest_len = len(line)

            for char_itr in range(len(line)):
                character = line[char_itr]
                if character not in chars:
                    char_to_index[character] = len(char_to_index.items())
                    chars.add(character)

        for itr in range(len(data)):
            pad_len = longest_len - len(data[itr])
            data[itr] = (PAD_CHAR * pad_len) + data[itr]

        

        data_X = [line[:-1] for line in data]
        data_Y = [line[-1] for line in data]

        apply_vocab(data_X, char_to_index)
        apply_vocab(data_Y, char_to_index)


        return (data_X, data_Y, char_to_index, longest_len)

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
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, X, Y, char_to_index, work_dir, longest_len):
        cuda_available = torch.cuda.is_available()
        print(f"Cuda check = {cuda_available}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        m = RNN_Model(input_size=len(char_to_index), output_size=len(char_to_index), hidden_dim=HIDDEN_DIM, n_layers=N_RNN_LAYERS)
        optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

        for epoch in range(1, N_EPOCHS + 1):
            print(f"Epoch: {epoch}")
            batch_X = [X[i:i+BATCH_SIZE] for i in range(0, len(X), BATCH_SIZE)]
            batch_Y = [Y[i:i+BATCH_SIZE] for i in range(0, len(Y), BATCH_SIZE)]
            for itr in range(len(batch_X) - 1):
                one_hot_matrix = get_features(batch_X[itr], char_to_index, longest_len)
                input_vec = torch.from_numpy(one_hot_matrix)
                output_vec = torch.Tensor(batch_Y[itr])
                self.train_batch(optimizer, device, m, batch_X[itr], batch_Y[itr])
            

    def train_batch(self, optimizer, device, m, X, Y):
        optimizer.zero_grad()
        X.to(device)
        Y.to(device)
        output, hidden = m(X)
        loss = F.cross_entropy(output, Y)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")



    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir, m):
        torch.save(m.state_dict(), work_dir)

    @classmethod
    def load(cls, work_dir):
        m = RNN_Model(*args, **{})
        m.load_state_dict(torch.load(work_dir))
        return m


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
        print('Training')
        model.run_train(X, Y, char_to_index, args.work_dir, longest_len)
        print('Saving model')
        model.save(args.work_dir, model)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
