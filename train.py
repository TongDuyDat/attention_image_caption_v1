import sys
from dataloder import data_loader
from config import *
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torch 
from model.attention import Encode, Decode
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
def train_epoch(dataloader, encoder, decoder, encode_optim, decode_optim, criterion, device):
    """Train the model for one epoch on the given dataset."""
    # Set the model to training mode (as opposed to evaluation mode).
    
    print("Training in device: ", device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.train()
    decoder.train()
    total = 0
    correct = 0
    loss_sum = 0
    for i, data in enumerate(tqdm(dataloader)):
        imgs, captions = data
        imgs = imgs.to(device)
        captions = captions.to(device)
        encode_optim.zero_grad()
        decode_optim.zero_grad()
        
        encode_out = encoder(imgs)
        # decode_hidden = decoder.reset_state(imgs.size(0))
        decode_outs, decode_hidden, attentions = decoder(encode_out, captions)
        loss = criterion(
            decode_outs.view(-1, decode_outs.size(-1)),
            captions.view(-1)
        )
        loss.backward()
        encode_optim.step()
        decode_optim.step()
        loss_sum += loss.item()
    return loss_sum/len(dataloader)

def validate(valloader, encoder, decoder, criterion, device, epoch, tokenizer):
    """Validate the model on the validation set and return the average loss obtained."""
    print("Validate the model on epoch {}".format(epoch))
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    val_loss = 0.0
    bleu_score_avg = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            imgs, captions = data
            imgs = imgs.to(device)
            captions = captions.to(device)    
            encode_out = encoder(imgs)
            # decode_hidden = decoder.reset_state(imgs.size(0))
            decode_outs, decode_hidden, attentions = decoder(encode_out, captions)
            loss = criterion(
                decode_outs.view(-1, decode_outs.size(-1)),
                captions.view(-1)
            )
            decode_score, decode_pre = torch.max(decode_outs, dim= -1)
            # bleu_torch  = bleu_score(captions, decode_pre)
            # print("Bleu torch: ", bleu_torch)
            captions_str = tokenizer.token2text(captions)
            decode_pre_str = tokenizer.token2text(decode_pre)
            bleu_nltk = corpus_bleu(captions_str, decode_pre_str,  smoothing_function= SmoothingFunction().method1)
            val_loss += loss.item()
            bleu_score_avg+= bleu_nltk
    val_avg_loss, val_bleu_score = val_loss/len(valloader), bleu_score_avg/len(valloader)
    print("Loss validate: {}, Bleu Score validate: {}".format(val_avg_loss, val_bleu_score))
    return val_avg_loss, val_bleu_score
        
def train(epochs, batch_size, lr):
    
    # Define the constant values
    embedding_dim = 256
    units = 512
    features_shape = 512
    attention_features_shape = 49
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Loading data 
    trainloader, valloader, testloader, tokenizer = data_loader(image_path, annotation_path, batch_size= 32)
    vocab_size = len(tokenizer.vocab)
    # Init model encoder and decoder 
    encoder = Encode(embedding_dim, embedding_dim)
    decoder = Decode(embedding_dim, units, vocab_size)
    
    # Init loss
    criterion = nn.CrossEntropyLoss()
   
    # Init optimizizer 
    encode_optim = optim.Adam(encoder.parameters(), lr = lr)
    decode_optim = optim.Adam(decoder.parameters(), lr = lr)
    
    
    # data test
    trainloader = valloader
    valloader = testloader
    
    valid_loss_save = sys.maxsize
    print("Start Training...")
    for epoch in range(epochs):
        print("Epoch {} : ".format(epoch+1))      
        train_loss = train_epoch(trainloader, encoder, decoder, encode_optim, decode_optim, criterion, device)
        print ("Train Loss : ", train_loss, end=" ")
        # Validation Phase
        with torch.no_grad():
            valid_loss, _ = validate(valloader, encoder, decoder, criterion, device, epoch, tokenizer)
            # Save model
            if valid_loss <= valid_loss_save:
                print("Save model in train/model/")
                torch.save(encoder.state_dict(), "train/model/best_encode.pt")
                torch.save(decoder.state_dict(), "train/model/best_decode.pt")
  
train(epochs= 10, batch_size= 64, lr= 0.001)
# def train_epoch
# def val_epoch
# def train
#      load data
#      calculate loss, 
#      save model
#      calculate matrics
