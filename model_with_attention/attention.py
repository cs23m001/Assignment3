import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
  """
      An attention mechanism that computes context vectors and attention weights.

      Attributes:
      ----------
      W_att : nn.Linear
          A linear layer to project the query.
      U_att : nn.Linear
          A linear layer to project the keys.
      V_att : nn.Linear
          A linear layer to compute the attention scores.

      Methods:
      -------
      forward(query, key):
          Computes the context vector and attention weights.
    """
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.W_att = nn.Linear(hidden_size, hidden_size)
    self.U_att = nn.Linear(hidden_size, hidden_size)
    self.V_att = nn.Linear(hidden_size, 1)

  def forward(self, query, key):
    # Computing the unnormalized attention score
    att_score = self.V_att(torch.tanh(self.W_att(query) + self.U_att(key)))
    att_score = att_score.squeeze(2).unsqueeze(1)

    # Normalized attention weights
    alpha = F.softmax(att_score, dim=-1)
    contex = torch.bmm(alpha, key)

    return contex, alpha


