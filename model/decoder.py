import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
        A vanilla decoder class for sequence-to-sequence models using different types of RNN cells.
        
        Attributes:
        ----------
        hidden_size : int
            The number of features in the hidden state of the RNN.
        embedding_size : int
            The size of the embedding vectors.
        cell_type : str
            The type of RNN cell to use ('rnn', 'lstm', or 'gru').
        dropout : float
            The dropout probability.
        bidirectional : bool
            Indicates whether the RNN is bidirectional.
        input_size : int
            The size of the input to the RNN cell, adjusted for bidirectional RNNs.
        num_layers : int
            The number of recurrent layers.
        embedding : nn.Embedding
            The embedding layer that maps input tokens to embedding vectors.
        output_layer : nn.Linear
            The linear layer that maps the RNN output to the output vocabulary size.
        decoder_cell : nn.Module
            The RNN cell (RNN, LSTM, or GRU) used in the decoder.

        Methods:
        -------
        forward(inputs, hidden, cell_state=None):
            Performs the forward pass of the decoder.
    """
    def __init__(self, output_size, hidden_size, embedding_size, num_layers = 1, dropout= 0.0, cell_type='rnn', bidirectional = 'no'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.dropout = dropout
        self.bidirectional = True if bidirectional == 'yes' else False
        self.decoder_cell = None
        self.input_size = embedding_size
        self.num_layers = num_layers
 
        self.embedding = nn.Embedding(output_size, embedding_size)

        self.output_layer = nn.Linear(self.hidden_size, output_size)

        if self.bidirectional:
            self.input_size = self.hidden_size * 2
            self.output_layer = nn.Linear(self.hidden_size * 2, output_size)

        # for RNN
        if self.cell_type == 'rnn':
            self.decoder_cell = nn.RNN(input_size = self.embedding_size, 
                                       hidden_size = self.hidden_size,
                                       num_layers = num_layers,
                                       batch_first=True,
                                       dropout = self.dropout,
                                       bidirectional = self.bidirectional)
        # For LSTM
        elif self.cell_type == 'lstm':
            self.decoder_cell = nn.LSTM(input_size = self.embedding_size, 
                                       hidden_size = self.hidden_size,
                                       num_layers = num_layers,
                                       batch_first=True,
                                       dropout = self.dropout,
                                       bidirectional = self.bidirectional)
        # For GRU
        elif self.cell_type == 'gru':
            self.decoder_cell = nn.GRU(input_size = self.embedding_size, 
                                       hidden_size = self.hidden_size,
                                       num_layers = num_layers,
                                       batch_first=True,
                                       dropout = self.dropout,
                                       bidirectional = self.bidirectional)   
        else:
            raise ValueError("Enter the correct cell_type.....")
        
    def forward(self, inputs, hidden, cell_state = None):
        embedded_output = self.embedding(inputs)
        embedded_output = F.relu(embedded_output)
        # print(embedded_output.shape)

        # For multilayer model taking the hidden state for the last layer
        # concaitinate for dimension matching
        if self.bidirectional:
            hidden = hidden[-1,:,:].unsqueeze(0)
            hidden = torch.cat([hidden]*self.num_layers*2, dim = 0)
        else:
            hidden = hidden[-1,:,:].unsqueeze(0)
            hidden = torch.cat([hidden]*self.num_layers, dim = 0)

        # Forward pass through the prefered cell
        if self.cell_type == 'lstm':
            # For cell state
            if self.bidirectional:
                cell_state = cell_state[-1,:,:].unsqueeze(0)
                cell_state = torch.cat([cell_state]*self.num_layers*2, dim = 0)
            else:
                cell_state = cell_state[-1,:,:].unsqueeze(0)
                cell_state = torch.cat([cell_state]*self.num_layers, dim = 0)

            output, (hidden_state, cell_state) = self.decoder_cell(embedded_output, (hidden, cell_state))
            output = self.output_layer(output)

            return output, (hidden_state, cell_state)
        else:    
            output, hidden_state = self.decoder_cell(embedded_output, hidden)
            output = self.output_layer(output)
        
            return output, hidden_state        
    

