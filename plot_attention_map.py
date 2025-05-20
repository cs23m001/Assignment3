import os
import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import torch
from corpus import Corpus
from model_with_attention import seq2seq_with_attention
from data_preprocessing import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_word(source, pred, corpus):
    """
        Convert tokenized sequences to words.

        Parameters:
        source (Tensor): Tokenized source sequences.
        target (Tensor): Tokenized target sequences.
        pred (Tensor): Tokenized predicted sequences.
        corpus (Corpus): Corpus object containing vocabulary mappings.

        Returns:
        tuple: Lists of source words, target words, and predicted words.
    """

    source_words = []
    pred_words = []

    for word in source.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(corpus.source_int2char[letter])

        temp_word = ''.join(temp_word)
        source_words.append(temp_word)

    for word in pred.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(corpus.target_int2char[letter])

        temp_word = ''.join(temp_word)
        pred_words.append(temp_word)    

    return source_words, pred_words

def get_attention_map(model, dataloader):
    """
        Get the attention map for the model predictions.

        This function computes the attention map for the model predictions on the given data.

        Parameters:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test data.

        Returns:
        tuple: Lists of source words, predicted words, and attention map.
    """

    c = Corpus(lang = 'hi', type = 'test')  

    for data in dataloader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs, _, attention_map = model(inputs)
        outputs = outputs.argmax(-1)

        outputs = outputs

        source, pred = get_word(inputs, outputs, c)
        break


    return source, pred, attention_map.detach().cpu().numpy()    


def plot_attention(input_word, predicted_word, attention_map, file_name = None):
  """
        Plot the attention heatmap.

        This function generates and displays the attention heatmap based on the given input word, predicted word,
        and attention map.

        Parameters:
        input_word (str): The input word.
        predicted_word (str): The predicted word.
        attention_map (numpy.ndarray): The attention map.
        file_name (str): Optional. The name of the file to save the plot.

        Returns:
        None
    """
  
  path = os.path.join(os.getcwd(), 'Attention_Heatmap', file_name)
  prop = fm.FontProperties(fname = os.path.join(os.getcwd(), 'Attention_Heatmap', 'Akshar-VariableFont_wght.ttf'))
  
  fig = plt.figure(figsize=(3, 3))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention_map)
  
  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + list(input_word), fontdict=fontdict, rotation=0)
  ax.set_yticklabels([''] + list(predicted_word), fontdict=fontdict,fontproperties=prop)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#   plt.show()

  plt.savefig(path)
  with wandb.init(entity="cs23m001-iit-m", project="ass3_att_heatmap", name=file_name):
        wandb.log({'Attention Heatmap' : wandb.Image(plt)})
  wandb.finish()    


# Loading the test dataloader
test_loader, input_size, output_size, max_seq_len = get_dataloader(lang = 'hi', type = 'test', batch_size = 256)

# Initializing the model
model = seq2seq_with_attention.Model(input_size = input_size, output_size = output_size, embedding_size = 128, hidden_size = 512,
                   num_encoder_layers = 3, num_decoder_layers = 1, cell_type = 'gru',
                   dropout = 0.3, bidirectional = 'yes', max_seq_len = max_seq_len)
model = model.to(device)  

# Loading the pretrained model
model.load_state_dict(torch.load('./trained_model/attention/attention_best_model.pth'))

# Collectiong the attention map
source_words, pred_words, attentions  = get_attention_map(model,test_loader)


for i in range(10):
    plot_attention(source_words[i], pred_words[i], attentions[i][1 : len(pred_words[i]), 1 : len(source_words[i])], f'heatmap_{i+1}')

