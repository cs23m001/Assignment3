# Transliteration System Using Recurrent Neural Network
In this assignment I have build a character lavel transliteration model using the recurrent neural network. I have also add Attention mechanism to improve the accuracy.

# Wandb Report Link
Please find the wandb report link for this project [here](https://api.wandb.ai/links/cs23m001-iit-m/hwxt9a9u)

# Library Used
The list of libraries used for this project are
    
 * Pytorch
 * Numpy
 * Pandas
 * tqdm
 * wandb
 * argparse
 * matplotlib

# Installation
The training and evaluation code requires above mentioned packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:
 * Create seperate conda environment

```bash
conda create -n assignment_3
conda activate assignment_3
```

 * Install Pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
`All the code in this experiment are trained and tested using gpu enabel system. Hence to run all the code properly a gpu enable machine is must.` If you find any dificulties then install pytorch version and corresponding cuda version accordingly.

 * Installing extra requirements The above command will download most of the necessary dependencies along with pytorch. For installing the rest of the packages run

 ```bash
 pip install requirements.txt
 ```

 * Download/Clone the github repository

 ```bash
git clone https://github.com/cs23m001/Assignment3.git
cd Assignment3
```

# Repository Deatils

The repository contains list of directories and python file. 
* `Attention_Heatmap` - Contains the attention heatmap of 9 test samples as .png format.
* `Dataset` - The dataset used for this assignment.
* `model` - Contains python file for valina (without attention) implementation.
    * `encoder.py` - The implementation of the valina encoder.
    * `decoder.py` - The implementaion of the vanila decoder model.
    * `seq2seq.py` - The sequence to sequence model for the transliteration task.
* `model_with_attention` - Contains python files for attention based implementation.
    * `attention.py` - The implementation of the Bahdanau Attention mechanism.
    * `encoder.py` - The encoder for the attention based model. It is exactly same as the vanila encoder.
    * `decoder.py` - Attention based decoder implementation.
    * `seq2seq_with_attention.py` - The attention based seqence to sequence model implementation.  
* `predictions_attention` - Contains the prediction for the attention based model.
    * `test_prediction.txt` - Contains the prediction on the test set. Each row in the text file contains first source word followed by target word followed by the prediction. 
* `predictions_vanila` - Contains the prediction for the vanila model.
    * `test_prediction.txt` - Contains the prediction on the test set. Each row in the text file contains first source word followed by target word followed by the prediction.   
* `trained_model` - Contain the pre-trained model weights trained on `hindi` dataset.
    * `attention` - Contain the model weights for the attention based model.
    * `vanila` - Contain the model weights for the attention based model.
* `validation` - Contain the text file for each epoch during training. Each text file is the prediction on the validation set. Each row of the text file contains first sorce word followed by target word followed by predicted word. The words are seperated by `-`.
    * `attention` - Contains epoch wise validation prediction for attention based model during training.
    * `vanila` - Contains epoch wise validation prediction for vanila model during training.
* `corpus.py` - Contains the python script responsible for creating the vocabulary, maximum seqience length and various other neccesary things for a specific language.
* `data_preprocessing.py` - Contains python script to create a torch dataset form the coupus and  return the dataloder.
* `evaluate_vanila.py` - Contains python script to evaluate the model on the test dataset.
* `evaluate_attention.py` - Contain python script to evaluate the attention based model on the test data.
* `plot_attention_map.py` - Contain the  python script to plot the attention heatmap for 9 test words.
* `train_vanila.py` - Python script to train the vanila sequence to seequence model on  any specific language.
* `train_attention.py` - Python script to train the attention based sequence to sequence model on any specific language.
* `wandb_attention.py` - Hyperparameter search using wandb for attention based model. 
* `wandb_vanila.py` - Hyperparameter search using wandb for vanila model. 

# How to Run
The dataset, pre-trained model will be there inside the respective directory. You need to just clone the repository and then you can run any file you want. Notice I have used `Hindi` language to train and evaluate the model. If you want to train the model on different language then pass that language as commanndline argument during training.

# Training
* In order train the vanila sequence to sequence model just  run 
```bash
python train_vanila.py --epochs 20 --batch_size 64 --learning_rate 0.0001 --language 'hi' --embedding_size 128 --hidden_size 512 --num_encoder_layers 1 --num_decoder_layers 4 --cell_type 'lstm' --dropout 0.4 --bidirectional 'yes' --save_model 'no'
```

This command will train the vanila model with the mentioned hyperparameter. This is default set of parameter obtained using the wandb search.
You can also run 
```bash
python train_vanila.py
```
which also train the  model using default parameter.

* If you want too train the attention based model, then simply run 
```bash
python train_attention.py --epochs 20 --batch_size 256 --learning_rate 0.001 --language 'hi' --embedding_size 128 --hidden_size 512 --num_encoder_layers 3 --num_decoder_layers 1 --cell_type 'gru' --dropout 0.3 --bidirectional 'yes' --save_model 'no'
```
This command will train the attention based model with the mentioned hyperparameter. This is default set of parameter obtained using the wandb search.
You can also run 
```bash
python train_attention.py
```
which also train the  model using default parameter.

* If you want to train on different language,  please change the `--language` parameter according to your choice. `Himdi` is the default language.
* `Command line argument  list` - 
    * `--epochs` - Number epoch the model will be trained.
    * `--batch_size` - Batch size for training.
    * `--learning_rate` - Learning rate for the optimizer.
    * `--language` - Language on whih   you want too train your model. Default `Hindi`
    * `--embedding_size` - Embedding dimension
    * `--hidden_size` - Hidden state size for the encoder and decoder module.
    * `--num_encoder_layers` - Number layers in the encoder module.
    * `--num_decoder_layers` - Number of layers in the decoder module.
    * `--cell_type` - Type of recurrent neural network you want to use for the encoder annd decoder RNN/LSTM/GRU
    * `--dropout` - The dropout percentage
    * `--bidirectional` - For bidirectional computation of the  model
    * `--save_model` - To save the trained model.

# Evaluation
In order to evaluate the moodel run 
* For vanila  model
```bash
python evaluate_vanila.py
```
This command will evaluate the vanila sequence to sequencce model on the test data and report the test accuracy

* For attention based model 
```bash
python evaluate_attention.py
```
This command will evaluate the attention based sequence to sequencce model on the test data and report the test accuracy.

* If you want to visuaalize the attention heatmap then run
```bash
python plot_attention_map.py
```
Notice every time  you run this command, it will showw you attention map for diifferent words as the each time it will fetch different wordds from randomly shuffeled batch.

# Acknowledgements
* [http://www.cse.iitm.ac.in/~miteshk/CS6910.html](http://www.cse.iitm.ac.in/~miteshk/CS6910.html)
* [https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c](https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c)