import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from corpus import Corpus

def get_dataloader(lang = 'hi', type = 'train', batch_size=64):
    """
        Creates a DataLoader for the given language and dataset type.

        Parameters:
        ----------
        lang : str, optional
            The language of the dataset (default is 'ben' for Bengali).
        type : str, optional
            The type of dataset ('train', 'test', etc.) (default is 'train').
        batch_size : int, optional
            The number of samples per batch to load (default is 64).

        Returns:
        -------
        DataLoader
            DataLoader object for the dataset.
        int
            Size of the source vocabulary.
        int
            Size of the target vocabulary.
        int
            Maximum sequence length for both source and target sequences.

    """
    # Initializzing the corpus
    c = Corpus(lang = lang, type = type)
    
    # Initialize matrices to hold tokenized source and target sequences
    source_matrix = np.ones((len(c.source_words), c.max_seq_len), dtype=np.int32)
    target_matrix = np.ones((len(c.target_words), c.max_seq_len), dtype=np.int32)
    

    # Tokenize and pad sequences
    for idx in range(len(c.source_words)):
        source_tokenize_words = c.source_tokenizer(c.source_words[idx])
        target_tokenize_words = c.target_tokenizer(c.target_words[idx])

        # Append end token to each sequence
        source_tokenize_words.append(c.source_vocab['<_end_>'])
        target_tokenize_words.append(c.target_vocab['<_end_>'])

        # Populate the matrices with tokenized words
        source_matrix[idx, :len(source_tokenize_words)] = source_tokenize_words
        target_matrix[idx, :len(target_tokenize_words)] = target_tokenize_words

    # Create TensorDataset from source and target matrices
    dataset = TensorDataset(torch.LongTensor(source_matrix), torch.LongTensor(target_matrix))  

    # Create a RandomSampler for the dataset
    data_sampler = RandomSampler(dataset) 

    # Create a dataloader
    data_loader = DataLoader(dataset, sampler = data_sampler, batch_size = batch_size)

    return data_loader, len(c.source_vocab), len(c.target_vocab), c.max_seq_len
