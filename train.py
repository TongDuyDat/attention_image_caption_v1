import sys
from dataloder import data_loader
from config import *
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torch 
from model.attention import Encode, Decode, Squen2Squen
from torchtext.data.metrics import bleu_score
from utils import plot_visual
from test import testing, testing_in_a_img
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
def train_epoch(dataloader, model, encode_optim, criterion, device, tokenizer):
    """Train the model for one epoch on the given dataset."""
    # Set the model to training mode (as opposed to evaluation mode).
    
    print("Training in device: ", device)
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)
    # encoder.train()
    # decoder.train()
    model.to(device)
    model.train()
    total = 0
    correct = 0
    loss_sum = 0
    bleu_score_avg = 0.0
    for i, data in enumerate(tqdm(dataloader)):
        imgs, captions = data
        imgs = imgs.to(device)
        captions = captions.to(device)
        encode_optim.zero_grad()
        # decode_optim.zero_grad()
        
        # encode_out = encoder(imgs)
        # decode_hidden = decoder.reset_state(imgs.size(0))
        decode_outs, decode_hidden, attentions = model(imgs, captions)
        loss = criterion(
            decode_outs.view(-1, decode_outs.size(-1)),
            captions.view(-1)
        )
        
        decode_outs = nn.functional.softmax(decode_outs, dim = -1)
        decode_score, decode_pre = torch.max(decode_outs, dim = -1)
        
        captions_str = tokenizer.token2text(captions)
        decode_pre_str = tokenizer.token2text(decode_pre)
        
        # score bleu 
        bleu_score_nltk = corpus_bleu(captions_str, decode_pre_str, smoothing_function= SmoothingFunction().method1)
        bleu_score_avg += bleu_score_nltk
        loss.backward()
        encode_optim.step()
        # decode_optim.step()
        loss_sum += loss.item()
    loss_avg, bleu_score_avg = loss_sum/len(dataloader), bleu_score_avg/len(dataloader)
    return loss_avg, bleu_score_avg

def validate(valloader, model, criterion, device, epoch, tokenizer):
    """Validate the model on the validation set and return the average loss obtained."""
    print("Validate the model on epoch {}".format(epoch))
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)
    # encoder.eval()
    # decoder.eval()
    model.to(device)
    model.train()
    val_loss = 0.0
    bleu_score_avg = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            imgs, captions = data
            imgs = imgs.to(device)
            captions = captions.to(device)    
            # encode_out = encoder(imgs)
            # decode_hidden = decoder.reset_state(imgs.size(0))
            decode_outs, decode_hidden, attentions = model(imgs, captions, 0)
            loss = criterion(
                decode_outs.view(-1, decode_outs.size(-1)),
                captions.view(-1)
            )
            decode_outs = nn.functional.softmax(decode_outs, dim = -1)
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Loading data 
    trainloader, valloader, testloader, tokenizer = data_loader(image_path, annotation_path, batch_size= batch_size)
    vocab_size = len(tokenizer.vocab)
    # Init model encoder and decoder 
    encoder = Encode(embedding_dim, embedding_dim)
    decoder = Decode(embedding_dim, units, vocab_size)
    model = Squen2Squen(encoder, decoder, max_length=tokenizer.max_length)
    # Init loss
    criterion = nn.CrossEntropyLoss(ignore_index= 0)
   
    # Init optimizizer 
    encode_optim = optim.Adam(model.parameters(), lr = lr)
    # decode_optim = optim.Adam(decoder.parameters(), lr = lr)
    
    
    # data test
    # trainloader = valloader
    # valloader = testloader
    
    valid_loss_save = sys.maxsize
    
    history_loss_train = []
    history_bleu_score_train = []
    
    history_loss_val = []
    history_bleu_score_val = []
    print("Start Training...")
    for epoch in range(epochs):
        print("Epoch {} : ".format(epoch+1))      
        train_loss, bleu_score_train = train_epoch(trainloader, model, encode_optim, criterion, device, tokenizer)
        print ("Train Loss : {} Bleu Score: {}".format(train_loss, bleu_score_train))
        # Validation Phase
        with torch.no_grad():
            valid_loss, bleu_score_val = validate(valloader, model, criterion, device, epoch, tokenizer)
            # Save model
            if valid_loss <= valid_loss_save:
                print("Save model in train/model/")
                torch.save(encoder.state_dict(), "train/model/best_encode.pt")
                torch.save(decoder.state_dict(), "train/model/best_decode.pt")
                valid_loss_save = valid_loss
        history_loss_train.append(train_loss)
        history_bleu_score_train.append(bleu_score_train)
        
        history_loss_val.append(valid_loss)
        history_bleu_score_val.append(bleu_score_val)
    plot_visual((history_loss_train, history_loss_val), (history_bleu_score_train, history_bleu_score_val))
    bleu = testing(testloader, model, tokenizer, device)
    # testing_in_a_img("D:/NCKH/ImageCaption/Dataset/Flickr8k_Dataset/Flicker8k_Dataset/17273391_55cfc7d3d4.jpg", encoder, decoder, tokenizer, device, visual= True)
    print("Bleu Score: ", bleu)
    
train(epochs= 10, batch_size= 256, lr= 0.01)
# def train_epoch
# def val_epoch
# def train
#      load data
#      calculate loss, 
#      save model
#      calculate matrics
