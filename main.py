from preprocess import *
from MLP import *

vocab_size = len(phoneme_to_ix)
embedding_dim = 100
hidden_dim = 256
model = PredictPastTense(vocab_size, embedding_dim, hidden_dim, max_length)

train(model, train_present, train_past, train_types_encoded, type_to_ix)
test(model, test_present, test_past, test_types_encoded, type_to_ix)
