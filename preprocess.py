#preprocess data
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
filepath = 'phonemized_forms.txt'




#preprocess helper
def preprocess_file_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]

    present_tense = []
    past_tense = []
    verb_types = []

    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) == 5:
            #get the present and past tense phonemic forms and verb type
            _, _, verb_type, present_phonemic, past_phonemic = parts
            present_tense.append(present_phonemic)
            past_tense.append(past_phonemic)
            verb_types.append(verb_type)

    return present_tense, past_tense, verb_types

present_tense, past_tense, verb_types = preprocess_file_data(filepath)



#create phoneme vocab
phoneme_vocab = Counter(phoneme for words in zip(present_tense, past_tense) for word in words for phoneme in word)
phoneme_vocab['<pad>'] = 0
phoneme_to_ix = {phoneme: ix for ix, phoneme in enumerate(phoneme_vocab)}
ix_to_phoneme = {ix: phoneme for phoneme, ix in phoneme_to_ix.items()}


print(phoneme_vocab)




def pad_sequences(sequences, max_length, pad_token):
    processed_sequences = []
    for seq in sequences:
      # Pad the sequence with pad_token up to max_length
      processed_seq = seq + [pad_token] * (max_length - len(seq))
      processed_sequences.append(processed_seq)
    return processed_sequences

# padding
# max length is the longest word in the dataset plus 2
max_length = max(len(word) for word in present_tense) + 2
print(max_length)
pad_token = phoneme_to_ix['<pad>']

present_indexed = [list(map(phoneme_to_ix.get, word)) for word in present_tense]
past_indexed = [list(map(phoneme_to_ix.get, word)) for word in past_tense]

present_processed = pad_sequences(present_indexed, max_length, phoneme_to_ix['<pad>'])
past_processed = pad_sequences(past_indexed, max_length, phoneme_to_ix['<pad>'])

present_tensors = torch.tensor(present_processed, dtype=torch.long)
past_tensors = torch.tensor(past_processed, dtype=torch.long)


# split into training and testing sets
#this is where data gets randomly shuffled
train_present, test_present, train_past, test_past, train_types, test_types = train_test_split(present_tensors, past_tensors, verb_types, test_size=0.2, shuffle=True) #ensures the training/test data are randomly shuffled

#to show that samples are randomized
print("\nSample from the testing set:")
for i in range(5):
    print(f"Present tense (test): {test_present[i]}, Past tense (test): {test_past[i]}, Verb type (test): {test_types[i]}")

#encode the verb types as tensors
type_to_ix = {'reg': 0, 'irreg': 1}
train_types_encoded = torch.tensor([type_to_ix[t] for t in train_types], dtype=torch.long)
test_types_encoded = torch.tensor([type_to_ix[t] for t in test_types], dtype=torch.long)

(train_present.shape, train_past.shape, train_types_encoded.shape), (test_present.shape, test_past.shape, test_types_encoded.shape)
