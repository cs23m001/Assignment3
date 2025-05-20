import torch
import torch.nn as nn
from corpus import Corpus
from model_with_attention import seq2seq_with_attention
from data_preprocessing import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(model, dataloader, criterion):
    """
        Validate the model on validation data.

        Parameters:
        model (nn.Module): The model to be validated.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.

        Returns:
        tuple: Validation accuracy and validation loss.
    """

    val_loss = 0
    total_sample = 0
    total_correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, _ = model(inputs)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            pred = outputs.argmax(-1)

            total_correct += (pred == targets).all(1).sum().item()
            total_sample += inputs.size(0)
            
            

            val_loss += loss

        val_accuracy = total_correct/total_sample   
        val_loss = val_loss/len(dataloader) 

        return  val_accuracy, val_loss

def get_word(source, target, pred, corpus):
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
    target_words = []
    pred_words = []

    for word in source.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(corpus.source_int2char[letter])

        temp_word = ''.join(temp_word)
        source_words.append(temp_word)

    for word in target.detach().cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(corpus.target_int2char[letter])

        temp_word = ''.join(temp_word)
        target_words.append(temp_word)

    for word in pred.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(corpus.target_int2char[letter])

        temp_word = ''.join(temp_word)
        pred_words.append(temp_word)    

    return source_words, target_words, pred_words

def save_pred(model, dataloader):
    """
        Save model predictions to a file.

        Parameters:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the validation data.
        iteration (int): Current epoch number.
    """

    c = Corpus(lang = 'hi', type = 'test')  
    source_list = []
    target_list = []
    pred_list = []

    for data in dataloader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs, _, _ = model(inputs)
        outputs = outputs.argmax(-1)

        outputs = outputs

        source, target, pred = get_word(inputs, targets, outputs, c)

        source_list.extend(source)
        target_list.extend(target)
        pred_list.extend(pred)

    with open(f"./predictions_attention/test_prediction.txt", 'w') as output:
        for (s,t,p) in zip(source_list, target_list,pred_list):
            temp = f"{s} - {t} - {p}"
            output.write((temp) + '\n')


# Defining the test loader
test_loader, input_size, output_size, max_seq_len = get_dataloader(lang = 'hi', type = 'test', batch_size = 256)

# Initializing the model
model = seq2seq_with_attention.Model(input_size = input_size, output_size = output_size, embedding_size = 128, hidden_size = 512,
                   num_encoder_layers = 3, num_decoder_layers = 1, cell_type = 'gru',
                   dropout = 0.3, bidirectional = 'yes', max_seq_len = max_seq_len)
model = model.to(device)  

# Loading the pretrained model
model.load_state_dict(torch.load('./trained_model/attention/attention_best_model.pth'))
criterion = nn.CrossEntropyLoss()

print(model)

# Calculating test accuracies
test_accuracy, test_loss = validation(model, test_loader, criterion)
print(f"Test Accuracy : {test_accuracy : 0.4f}")

# Saving Pridiction
# save_pred(model, test_loader)


