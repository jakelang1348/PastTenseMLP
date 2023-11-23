import re
import json
import sys
import collections
import os
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from preprocess import *




class PredictPastTense(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_length):
        super(PredictPastTense, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_to_hidden1 = nn.Linear(max_length * embedding_dim, hidden_dim)

        #second hidden layer added for testing
        self.hidden1_to_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, max_length * vocab_size)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        hidden1 = self.relu(self.input_to_hidden1(flattened))
        #second hidden layer added for testing
        hidden2 = self.relu(self.hidden1_to_hidden2(hidden1))

        output = self.hidden_to_output(hidden2)
        output = output.view(-1, self.max_length, self.vocab_size)
        return output

lr = 0.001

def train(model, train_present, train_past, train_types_encoded, type_to_ix):
    #batching
    dataset = torch.utils.data.TensorDataset(train_present, train_past, train_types_encoded)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    #loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #loop
    num_to_train = 50
    for training_iteration in range(num_to_train):
        model.train()
        total_loss = 0
        total_correct_reg = 0
        total_correct_irreg = 0
        total_reg = 0
        total_irreg = 0
        for in_batch, target_batch, types_batch in data_loader:
            #get predicted output
            out_batch = model(in_batch)
            #compute loss
            loss = criterion(out_batch.view(-1, model.vocab_size), target_batch.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #get loss
            total_loss += loss.item()

            #calculate accuracies for regular and irregular verbs
            _, predicted = torch.max(out_batch, 2)
            correct = predicted == target_batch
            for i, correct_tensor in enumerate(correct):
                verb_type = types_batch[i].item()
                if verb_type == type_to_ix['reg']:
                    total_correct_reg += correct_tensor.all().item()
                    total_reg += 1
                elif verb_type == type_to_ix['irreg']:
                    total_correct_irreg += correct_tensor.all().item()
                    total_irreg += 1

        #print average loss and accuracies for the iteration
        print(f'Iteration {training_iteration+1}/{num_to_train}, Loss: {total_loss/len(data_loader)}')
        if total_reg > 0:
            print(f'Regular Verb Accuracy: {total_correct_reg/total_reg:.4f}')
        if total_irreg > 0:
            print(f'Irregular Verb Accuracy: {total_correct_irreg/total_irreg:.4f}')


def test(model, test_present, test_past, test_types_encoded, type_to_ix):
    model.eval()
    total_correct_reg = 0
    total_correct_irreg = 0
    total_reg = 0
    total_irreg = 0
    incorrect_verbs = []

    with torch.no_grad():
        for i in range(len(test_present)):
            input_tensor = test_present[i].unsqueeze(0)
            target_tensor = test_past[i]
            verb_type = test_types_encoded[i].item()

            #keep track of num tested for each verb type
            if verb_type == type_to_ix['reg']:
              total_reg += 1
            elif verb_type == type_to_ix['irreg']:
              total_irreg += 1

            output_tensor = model(input_tensor)
            predicted_tensor = output_tensor.argmax(dim=2).squeeze(0)


            if torch.equal(predicted_tensor, target_tensor):
                if verb_type == type_to_ix['reg']:
                    total_correct_reg += 1
                elif verb_type == type_to_ix['irreg']:
                    total_correct_irreg += 1
            else:
              # If incorrect, decode the tensors to phonemes
                input_word = ''.join([ix_to_phoneme[ix.item()] for ix in input_tensor[0] if ix != phoneme_to_ix['<pad>']])
                predicted_word = ''.join([ix_to_phoneme[ix.item()] for ix in predicted_tensor if ix != phoneme_to_ix['<pad>']])
                correct_word = ''.join([ix_to_phoneme[ix.item()] for ix in target_tensor if ix != phoneme_to_ix['<pad>']])
                incorrect_verbs.append((input_word, predicted_word, correct_word))


    print("Incorrect predictions:")
    for original, predicted, correct in incorrect_verbs[:5]:  # Adjust the number as needed
        print(f"Original: {original}, Predicted: {predicted}, Correct: {correct}")

    #accuracy
    if total_reg > 0:
        reg_accuracy = total_correct_reg / total_reg
        print(f'Regular Verb Accuracy: {reg_accuracy:.4f}')
    if total_irreg > 0:
        irreg_accuracy = total_correct_irreg / total_irreg
        print(f'Irregular Verb Accuracy: {irreg_accuracy:.4f}')
    overall_accuracy = (total_correct_reg + total_correct_irreg) / (total_reg + total_irreg)
    print('Overall Accuracy on testing:', overall_accuracy)
    return reg_accuracy, irreg_accuracy, overall_accuracy
