import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
        An encoder class for sequence-to-sequence models using different types of RNN cells.

        Attributes:
        ----------
        hidden_size : int
            The number of features in the hidden state of the RNN.
        cell_type : str
            The type of RNN cell to use ('rnn', 'lstm', or 'gru').
        dropout : float
            The dropout probability.
        bidirectional : bool
            Indicates whether the RNN is bidirectional.
        embedding : nn.Embedding
            The embedding layer that maps input tokens to embedding vectors.
        encoder_cell : nn.Module
            The RNN cell (RNN, LSTM, or GRU) used in the encoder.

        Methods:
        -------
        forward(inputs):
            Performs the forward pass of the encoder.
    """

    def __init__(self, input_size, hidden_size, embedding_size, num_layers = 1, dropout= 0.0, cell_type='lstm', bidirectional = 'no'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.dropout = dropout
        self.bidirectional = True if bidirectional=='yes' else False
        self.encoder_cell = None

        self.embedding = nn.Embedding(input_size, embedding_size)

        # Defining the cell type
        if self.cell_type == 'rnn':
            self.encoder_cell = nn.RNN(input_size = embedding_size, 
                                       hidden_size = self.hidden_size,
                                       num_layers = num_layers,
                                       batch_first=True,
                                       dropout = self.dropout,
                                       bidirectional = self.bidirectional)
        elif self.cell_type == 'lstm':
            self.encoder_cell = nn.LSTM(input_size = embedding_size, 
                                       hidden_size = self.hidden_size,
                                       num_layers = num_layers,
                                       batch_first=True,
                                       dropout = self.dropout,
                                       bidirectional = self.bidirectional)
        elif self.cell_type == 'gru':
            self.encoder_cell = nn.GRU(input_size = embedding_size, 
                                       hidden_size = self.hidden_size,
                                       num_layers = num_layers,
                                       batch_first=True,
                                       dropout = self.dropout,
                                       bidirectional = self.bidirectional)   
        else:
            raise ValueError("Enter the correct cell_type.....")   

    # Forward pass through the network
    def forward(self, inputs):
        embedded_out = self.embedding(inputs)
        
        if self.cell_type == 'lstm':
            output, (hidden_state, cell_state) = self.encoder_cell(embedded_out) 

            return output, hidden_state,  cell_state
        else:
            output, hidden_state = self.encoder_cell(embedded_out) 
            
            return output, hidden_state        

