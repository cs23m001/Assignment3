import os
import torch
import argparse
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from data_preprocessing import get_dataloader
from model import seq2seq

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Training loop for each epoch
def train_epoch(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    total_correct = 0
    total_sample = 0
    # Iterating through dataset
    for data in tqdm(dataloader):
        inputs , targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        # Getting logits
        outputs, _, _ = model(inputs, targets)

        # Loss computation
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        # Backwaard pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss

        pred = outputs.argmax(-1)
        
        total_correct += (pred == targets).all(1).sum().item()
        total_sample += inputs.size(0)

    epoch_accuracy = total_correct/total_sample  
    epoch_loss =  epoch_loss/len(dataloader) 

    return  epoch_accuracy, epoch_loss

# Function for validataion
def validation(model, dataloader, criterion):
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
    
def train(config = None):

  with wandb.init(config=config):

    config = wandb.config
    wandb.run.name = wandb.run.name = 'cell_' + str(config.cell_type) + '-embd_s_' + str(config.embedding_size) + '-hiddn_s_' + str(
        config.hidden_size) + '-e_layer_' + str(config.num_encoder_layers) + '-d_layer_' + str(config.num_decoder_layers) + '-bs_' + str(config.batch_size) + '-lr_' + str(
                config.learning_rate) + '-dr_' + str(config.dropout) + '-bdir_' + str(config.bidirectional)

    # Train and test dataloader
    train_loader, input_size, output_size, max_seq_len = get_dataloader(lang = 'hi', type = 'train', batch_size = config.batch_size)
    val_loader, _, _, _ = get_dataloader(lang = 'hi', type = 'dev', batch_size = config.batch_size)

    # Defining model
    model = seq2seq.Model(input_size = input_size, output_size = output_size, embedding_size = config.embedding_size, hidden_size = config.hidden_size,
                    num_encoder_layers = config.num_encoder_layers, num_decoder_layers = config.num_decoder_layers, cell_type = config.cell_type,
                    dropout = config.dropout, bidirectional = config.bidirectional, max_seq_len = max_seq_len)
    model = model.to(device)

    print(model)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Main training loop
    for epoch in range(config.epochs):
      
      print(f"Epoch {epoch+1}/{config.epochs} : ")

      # Train metrics and validation metrics
      train_accuracy, train_loss = train_epoch(model, train_loader, optimizer, criterion)
      val_accuracy, val_loss = validation(model, val_loader, criterion)

      print(f"Train Accuracy : {train_accuracy} \t Train Loss : {train_loss} \t Validation Accuracy : {val_accuracy} \t Validation Loss : {val_loss}")

      # Logging to wandb
      wandb.log( {"training_accuracy" : train_accuracy, "training_loss" : train_loss,
                        "validation_accuracy" : val_accuracy, "validation_loss" : val_loss} )

    wandb.run.save()

  # del model
  # torch.cuda.empty_cache()    



if __name__ == "__main__":
  metric = {
              'name' : 'validation_accuracy',
              'goal' : 'maximize'
          }
  parameters = {
                    'epochs' : { 'values' : [15] },
                    'batch_size' : { 'values' : [32, 64, 128, 256] },
                    'learning_rate' : { 'values' : [0.01, 0.001, 0.0001, 0.003, 0.0003] },
                    'embedding_size' : { 'values' : [32, 64, 128, 256, 512] },
                    'hidden_size' : { 'values' : [32, 64, 128, 256, 512] },
                    'num_encoder_layers' : { 'values' : [ 1, 2, 3, 4 ] },
                    'num_decoder_layers' : { 'values' : [ 1, 2, 3, 4 ] },
                    'cell_type' : { 'values' : ['rnn', 'lstm', 'gru'] },
                    'dropout' : { 'values' : [0.0, 0.1, 0.2, 0.3, 0.4] },
                    'bidirectional' : { 'values' : ['yes', 'no'] },
              }

  sweep_config = dict()
  sweep_config['name'] = 'batch_size'
  sweep_config['method'] = 'grid'
  sweep_config['metric'] = metric
  sweep_config['parameters'] = parameters
  #config = parse_argument()
  wandb.login(key="xxxxx")
  sweep_id = wandb.sweep(sweep_config, entity="xxx", project="xxx")
  wandb.agent(sweep_id, function=train)
  wandb.finish()  

