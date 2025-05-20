import os
import pandas as pd
import numpy as np
import torch

root_path  = 'path to your dataset root directory .... exp. /home/../../dakshina_dataset_v1.0/'

class Corpus:
    """
        A class to process and handle the corpus for sequence-to-sequence modeling.

        Attributes:
        ----------
        lang : str
            The language of the corpus.
        type : str
            The type of dataset ('train', 'test', etc.).
        source_words : list
            List of source words in the corpus.
        target_words : list
            List of target words in the corpus.
        source_vocab : dict
            Vocabulary for the source language with character to index mapping.
        target_vocab : dict
            Vocabulary for the target language with character to index mapping.
        source_int2char : dict
            Mapping from indices to characters for the source language.
        target_int2char : dict
            Mapping from indices to characters for the target language.
        max_seq_len : int
            Maximum sequence length for both source and target words.
        source_max_seq_len : int
            Maximum sequence length for source words.
        target_max_seq_len : int
            Maximum sequence length for target words.

        Methods:
        -------
        read_file(path):
            Reads the CSV file and extracts source and target words.
        create_vocab(lang):
            Creates vocabulary for the source and target languages.
        source_tokenizer(word):
            Tokenizes a source word into its corresponding indices.
        target_tokenizer(word):
            Tokenizes a target word into its corresponding indices.
    """

    def __init__(self, lang = 'hi', type = 'train'):
        self.lang = lang
        self.type = type
        self.source_words = None
        self.target_words = None

        self.source_vocab = None
        self.target_vocab = None

        self.source_int2char = None
        self.target_int2char = None

        self.max_seq_len = 0
        self.source_max_seq_len = 0
        self.target_max_seq_len = 0

        # Collectiong all the source and target words
        # self.source_words , self.target_words = self.read_file(os.path.join(root_path, f"{self.lang}/{self.lang}_{self.type}.csv"))
        self.source_words , self.target_words = self.read_file(os.path.join(root_path, self.lang, 'lexicons', f"{self.lang}.translit.sampled.{self.type}.tsv"))

        # Creating the vocabulary
        self.create_vocab(self.lang)

        # Computing the max sequence length
        self.max_seq_len = max(self.source_max_seq_len, self.target_max_seq_len) + 1

    # Function for reading the csv file
    def read_file(self, path):
        words = pd.read_csv(path, sep='\t')
        source_words = words.iloc[:,1].tolist()
        target_words = words.iloc[:,0].tolist()

        source_words = [str(word) for word in source_words]
        target_words = [str(word) for word in target_words]

        return source_words, target_words  

    def create_vocab(self, lang):
        total_source_words = []
        total_target_words = []
        for file in (os.listdir(os.path.join(root_path, lang, 'lexicons'))):
            source_words, target_words = self.read_file(os.path.join(root_path, lang, 'lexicons', file))
            
            total_source_words.extend(source_words)
            total_target_words.extend(target_words)
            

        #Finding the max sequence length for source and target
        self.source_max_seq_len = max([len(word) for word in total_source_words])
        self.target_max_seq_len = max([len(word) for word in total_target_words])


        source_vocab = set(''.join(total_source_words)) 
        target_vocab = set(''.join(total_target_words)) 


        # Vocabulary as dictionary and inserting the start and end token
        self.source_int2char = {i+2 : char for i,char in enumerate(sorted(source_vocab))}
        self.source_int2char[0] = '<_start_>'
        self.source_int2char[1] = '<_end_>'

        # For int to character conversion
        self.target_int2char = {i+2 : char for i,char in enumerate(sorted(target_vocab))}
        self.target_int2char[0] = '<_start_>'
        self.target_int2char[1] = '<_end_>'

        self.source_vocab = {char : i+2 for i,char in enumerate(sorted(source_vocab))}   
        self.target_vocab = {char : i+2 for i,char in enumerate(sorted(target_vocab))}
        self.source_vocab['<_start_>'] = 0
        self.source_vocab['<_end_>'] = 1
        self.target_vocab['<_start_>'] = 0
        self.target_vocab['<_end_>'] = 1   

    # Function for source tokenization
    def source_tokenizer(self, word):
        return [self.source_vocab[l] for l in word]  

    # Function for target tokenization
    def target_tokenizer(self, word):
        return [self.target_vocab[l] for l in word] 

