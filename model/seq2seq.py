import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    """
        A sequence-to-sequence model that uses an encoder-decoder architecture with various RNN cells.

        Attributes:
        ----------
        cell_type : str
            The type of RNN cell to use ('rnn', 'lstm', or 'gru').
        max_sequence_len : int
            The maximum sequence length for the decoder.
        encoder : Encoder
            The encoder part of the model.
        decoder : Decoder
            The decoder part of the model.
        output : nn.Linear
            The linear layer that maps the hidden state to the output vocabulary size.

        Methods:
        -------
        forward(inputs, targets=None):
            Performs the forward pass of the model.
    """

    def __init__(self, input_size, output_size, embedding_size, hidden_size, num_encoder_layers, num_decoder_layers,
                cell_type, dropout, bidirectional, max_seq_len):
        super(Model, self).__init__()
        self.cell_type = cell_type
        self.max_sequence_len = max_seq_len
        
        # Encoder Model
        self.encoder = Encoder(input_size = input_size, hidden_size = hidden_size,
                               embedding_size = embedding_size, num_layers = num_encoder_layers, dropout = dropout,
                               cell_type = cell_type, bidirectional = bidirectional).to(device)

        # Decoder Model
        self.decoder = Decoder( output_size = output_size, hidden_size = hidden_size,
                               embedding_size = embedding_size, num_layers = num_decoder_layers, dropout= dropout,
                               cell_type=cell_type, bidirectional = bidirectional).to(device)
        
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs, targets = None):
        if self.cell_type == 'lstm':
            encoder_output, encoder_hidden, encoder_cell = self.encoder(inputs)
            decoder_cell = encoder_cell
        else:    
            encoder_output, encoder_hidden = self.encoder(inputs)

        batch_size = encoder_output.size(0)
        # Encoder hidden state is assign as the decoder initial hidden state
        decoder_hidden = encoder_hidden
        # Assigning the <_start_> as the first decoder inputs
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(0).to(device)
        # Array to store the generated output   
        decoder_outputs = []
        
        # Generating one letter at a time
        for i in range(self.max_sequence_len):
            if self.cell_type == 'lstm':
                decoder_output, (decoder_hidden, decoder_cell)  = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            else:    
                decoder_output, decoder_hidden  = self.decoder(decoder_input, decoder_hidden)

            decoder_outputs.append(decoder_output)

            # Teacher Forcing
            if targets is not None:
                decoder_input = targets[:,i].unsqueeze(1)
            else:    
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden, None    
    