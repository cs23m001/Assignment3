import os
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from data_preprocessing import get_dataloader
from model import seq2seq
from corpus import Corpus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, optimizer, criterion):
    """
        Train the model for one epoch.

        Parameters:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.

        Returns:
        tuple: Training accuracy and training loss for the epoch.
    """
    epoch_loss = 0
    total_correct = 0
    total_sample = 0
    # Iterating over the dataset
    for data in tqdm(dataloader):
        inputs , targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs, _, _ = model(inputs, targets)

        # Loss Computation
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss

        pred = outputs.argmax(-1)
        
        # Word lavel accuracy computation
        total_correct += (pred == targets).all(1).sum().item()
        total_sample += inputs.size(0)

    epoch_accuracy = total_correct/total_sample  
    epoch_loss =  epoch_loss/len(dataloader) 

    return  epoch_accuracy, epoch_loss

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

            # Word lavel accuracy
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
    
def save_pred(model, c, dataloader, iteration):
    """
        Save model predictions to a file.

        Parameters:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the validation data.
        iteration (int): Current epoch number.
    """ 
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

    with open(f"./validation/vanila/epoch_{iteration}.txt", 'w') as output:
        for (s,t,p) in zip(source_list, target_list,pred_list):
            temp = f"{s} - {t} - {p}"
            output.write((temp) + '\n')




def train(config):

    # Defining the dataloader
    train_loader, input_size, output_size, max_seq_len = get_dataloader(lang = config.language, type = 'train', batch_size = config.batch_size)
    val_loader, _, _, _ = get_dataloader(lang = config.language, type = 'dev', batch_size = config.batch_size)

    # Corpus definition
    cor = Corpus(lang = config.language, type = 'dev') 

    # Model Definition
    model = seq2seq.Model(input_size = input_size, output_size = output_size, embedding_size = config.embedding_size, hidden_size = config.hidden_size,
                   num_encoder_layers = config.num_encoder_layers, num_decoder_layers = config.num_decoder_layers, cell_type = config.cell_type,
                   dropout = config.dropout, bidirectional = config.bidirectional, max_seq_len = max_seq_len)
    model = model.to(device)
    
    print(model)

    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    max_val = 0.0

    for epoch in range(config.epochs):

        print(f"Epoch {epoch+1}/{config.epochs} : ")

        # Epoch wise training accuracy and loss
        train_accuracy, train_loss = train_epoch(model, train_loader, optimizer, criterion)
        # Eppochwise validation
        val_accuracy, val_loss = validation(model, val_loader, criterion)
        
        print(f"Train Accuracy : {train_accuracy:0.4f} \t Train Loss : {train_loss :0.4f} \t Validation Accuracy : {val_accuracy :0.4f} \t Validation Loss : {val_loss :0.4f}")

        # Saving the validation prediction
        save_pred(model, cor, val_loader, epoch+1)

        # Saving the model
        if config.save_model == 'yes':
            if val_accuracy > max_val:
                print("Validation Accuracy increased. Saving Model Parameters...")
                torch.save(model.state_dict(), './trained_model/vanila/vanila_best_model.pth')   
                max_val = val_accuracy 
 


def parse_argument():
    parser = argparse.ArgumentParser(description='Arguments for training')
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--language", type=str, default='hi')
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--cell_type", type=str, default='lstm')
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=str, default='yes')
    parser.add_argument("--save_model", type=str, default='no')

    return parser.parse_args()



if __name__ =="__main__":
    config = parse_argument()
    train(config)