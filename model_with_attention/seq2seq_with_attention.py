import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import DecoderWithAttention
from .encoder import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
  """
      A sequence-to-sequence model with attention mechanism using an encoder-decoder architecture.

      Attributes:
      ----------
      cell_type : str
          The type of RNN cell to use ('rnn', 'lstm', or 'gru').
      max_seq_len : int
          The maximum sequence length for the decoder.
      num_encoder_layer : int
          The number of layers in the encoder.
      num_decoder_layer : int
          The number of layers in the decoder.
      bidirectional : str
          Indicates whether the RNN is bidirectional ('yes' or 'no').
      encoder : Encoder
          The encoder part of the model.
      decoder : DecoderWithAttention
          The decoder part of the model with attention mechanism.

      Methods:
      -------
      forward(inputs, targets=None):
          Performs the forward pass of the model.
  """
  
  def __init__(self, input_size, output_size, embedding_size, hidden_size, num_encoder_layers, num_decoder_layers,
                cell_type, dropout, bidirectional, max_seq_len):
    super(Model, self).__init__()
    self.cell_type = cell_type
    self.max_seq_len = max_seq_len
    self.num_encoder_layer = num_encoder_layers
    self.num_decoder_layer = num_decoder_layers
    self.bidirectional = bidirectional

    # Encoder model
    self.encoder = Encoder(input_size = input_size, hidden_size = hidden_size,
                               embedding_size = embedding_size, num_layers = self.num_encoder_layer, dropout = dropout,
                               cell_type = cell_type, bidirectional = self.bidirectional)

    # Decoder Model
    self.decoder = DecoderWithAttention(output_size = output_size, hidden_size = hidden_size,
                               embedding_size = embedding_size, num_layers = self.num_decoder_layer, dropout= dropout,
                               cell_type=cell_type, bidirectional = self.bidirectional)

  def forward(self, inputs, targets=None):
    if self.cell_type == 'lstm':
      encoder_output, encoder_hidden, encoder_cell = self.encoder(inputs)
      decoder_cell = encoder_cell
    else:
      encoder_output, encoder_hidden = self.encoder(inputs)
 
    batch_size = encoder_output.size(0)

    # Assigning the encodeer hidden state as the decoder initial hidden state
    decoder_hidden = encoder_hidden
    
    # Setting the decoder first input as <_start_> token
    decoder_input = torch.empty(batch_size, 1, dtype = torch.long).fill_(0).to(device)
    decoder_outputs = []
    attentions = []

    # Degerating the output one leter at a time
    for i in range(self.max_seq_len):
      
      if self.cell_type == 'lstm':
        decoder_output, decoder_hidden, decoder_cell, alpha = self.decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_output,
                                                                             decoder_cell)
      else:
        decoder_output, decoder_hidden, alpha = self.decoder(decoder_input,
                                                               decoder_hidden,
                                                               encoder_output)

      # Decoder output
      decoder_outputs.append(decoder_output)
      # Attention weights
      attentions.append(alpha)

      # Teacher Forcing
      if targets is not None:
        decoder_input = targets[:,i].unsqueeze(1)
      else:    
        _, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze(-1).detach()

    decoder_outputs = torch.cat(decoder_outputs, dim=1)
    attentions = torch.cat(attentions, dim = 1)
    # decoder_outputs = F.log_softmax(decoder_outputs, dim = -1)

    return decoder_outputs, decoder_hidden, attentions
  
