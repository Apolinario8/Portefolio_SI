import numpy as np


class OneHotEncoder():

    def __init__(self, padder, max_length) -> None:
        self.padder = padder
        self.max_length = max_length

        self.alphabet = None
        self.char_to_index = None
        self.index_to_char = None


    def fit(self, data):
        self.alphabet = sorted(set(''.join(data).lower() + self.padder))

        self.char_to_index = {char: i for i, char in enumerate(self.alphabet)}
        self.index_to_char = {i: char for i, char in enumerate(self.alphabet)}

        if self.max_length is None:
            self.max_length = max(len(seq) for seq in data)


    def transform(self, data):
        data = [seq[:self.max_length].ljust(self.max_length, self.padder) for seq in data]

        one_hot_encoded = []
        for seq in data:
            one_hot_matrix = np.zeros((self.max_length, len(self.alphabet)))
            for i, char in enumerate(seq):
                one_hot_matrix[i, self.char_to_index[char]] = 1
            one_hot_encoded.append(one_hot_matrix)

        return one_hot_encoded
    

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    

    def inverse_transform(self, one_hot_encoded):
        decoded_sequences = []
        for one_hot_matrix in one_hot_encoded:
            decoded_sequence = ''.join(self.index_to_char[np.argmax(row)] for row in one_hot_matrix)
            decoded_sequences.append(decoded_sequence.rstrip(self.padder))
        return decoded_sequences
