import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention


class DecoderWithAttention(nn.Module):
  """
      A decoder with attention mechanism for sequence-to-sequence models.

      Attributes:
      ----------
      hidden_size : int
          The number of features in the hidden state.
      num_layers : int
          The number of recurrent layers.
      embedding_size : int
          The size of the embedding vectors.
      cell_type : str
          The type of RNN cell to use ('rnn', 'lstm', or 'gru').
      bidirectional : bool
          Indicates whether the RNN is bidirectional.
      dropout : nn.Dropout
          The dropout layer.
      embedding : nn.Embedding
          The embedding layer that maps output tokens to embedding vectors.
      output_layer : nn.Linear
          The linear layer that maps the hidden state to the output vocabulary size.
      attention : Attention
          The attention mechanism.
      decoder_input_size : int
          The size of the input to the decoder cell.
      decoder_cell : nn.Module
          The RNN cell (RNN, LSTM, or GRU) used in the decoder.

      Methods:
      -------
      forward(inputs, hidden, encoder_output, cell_state=None):
          Performs the forward pass of the decoder with attention.
  """
  def __init__(self, output_size, hidden_size, embedding_size, num_layers = 1, dropout = 0.0, cell_type = 'rnn', bidirectional = 'no'):
    super(DecoderWithAttention, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.embedding_size = embedding_size
    self.cell_type = cell_type
    self.bidirectional = False if bidirectional=='no' else True
    self.decoder_cell = None

    self.dropout = nn.Dropout(p = dropout)
    self.embedding = nn.Embedding(output_size, embedding_size)
    self.output_layer = nn.Linear(self.hidden_size, output_size)
    # Attention Module
    self.attention = Attention(self.hidden_size)
    self.decoder_input_size = self.hidden_size + self.embedding_size

    # Adjusting the shape for bidirectional model
    if self.bidirectional:
      self.decoder_input_size = (self.hidden_size * 2 + self.embedding_size)
      self.attention = Attention(self.hidden_size * 2)
      self.output_layer = nn.Linear(self.hidden_size * 2, output_size)

    if self.cell_type == 'rnn':
      self.decoder_cell = nn.RNN(input_size = self.decoder_input_size,
                                 hidden_size = self.hidden_size,
                                 num_layers = self.num_layers,
                                 batch_first = True,
                                 dropout = dropout,
                                 bidirectional = self.bidirectional
                                 )
    elif self.cell_type == 'lstm':
      self.decoder_cell = nn.LSTM(input_size = self.decoder_input_size,
                                 hidden_size = self.hidden_size,
                                 num_layers = self.num_layers,
                                 batch_first = True,
                                 dropout = dropout,
                                 bidirectional = self.bidirectional
                                 )
    elif self.cell_type == 'gru':
      self.decoder_cell = nn.GRU(input_size = self.decoder_input_size,
                                 hidden_size = self.hidden_size,
                                 num_layers = self.num_layers,
                                 batch_first = True,
                                 dropout = dropout,
                                 bidirectional = self.bidirectional
                                )
    else:
      raise ValueError("Select correct cell_type for decoder...")

  def forward(self, inputs, hidden, encoder_output, cell_state = None):
    # Embedding
    embedded_output = self.embedding(inputs)
    embedded_output = self.dropout(embedded_output)

    # For multilayer and bidirectional takeing the hidden state of the last 
    # layer and concatinate them for shape matching
    if self.bidirectional:
      query = torch.cat((hidden[-1,:,:], hidden[-2,:,:]), dim = 1)
      query = query.unsqueeze(0)
      hidden = hidden[-1,:,:].unsqueeze(0)
      hidden = torch.cat([hidden]*self.num_layers*2, dim = 0)
    else:
      query = hidden[-1,:,:].unsqueeze(0)  
      hidden = hidden[-1,:,:].unsqueeze(0)
      hidden = torch.cat([hidden]*self.num_layers, dim = 0)
    
    # Converting to Batch,Seq_len,Hidden_size shape
    query = query.permute(1,0,2)
    # Contex vector and the attention score
    context, alpha = self.attention(query, encoder_output)
    # Concatinating the  context vector with the decoder current input
    decoder_input = torch.cat((embedded_output, context), dim = 2)

    # Forward pass through the chossen cell type
    if self.cell_type == 'lstm':
      # Handling the shape missmatch issue
      if self.bidirectional:
        cell_state = cell_state[-1,:,:].unsqueeze(0)
        cell_state = torch.cat([cell_state]*self.num_layers*2, dim = 0)
      else:
        cell_state = cell_state[-1,:,:].unsqueeze(0)
        cell_state = torch.cat([cell_state]*self.num_layers, dim = 0) 
      
      output, (hidden_state, cell_state) = self.decoder_cell(decoder_input, (hidden, cell_state))
      
      output = self.output_layer(output)
      return output, hidden_state, cell_state, alpha

    else:
      output, hidden_state = self.decoder_cell(decoder_input, hidden)
      output = self.output_layer(output)

      return output, hidden_state, alpha


