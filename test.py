import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from config import *
from dataloder import data_loader
from model.vgg16_base import VGG16
from PIL import Image
from utils import visual_attentions
trainloader, valloader, testloader, tokenizer = data_loader(image_path, annotation_path, batch_size= 32)
vgg16 = VGG16()
def testing(data_loader, encoder, decoder, tokenizer, device):
    '''Testing phase'''
    encoder.eval()
    decoder.eval()
    total_test_bleu = 0.0
    
    with torch.no_grad():
        for img, cap in tqdm(data_loader):
            img = img.to(device)
            cap = cap.to(device)
            encoder_out = encoder(img)
            decode_out, decode_hidden, attetions = decoder(encoder_out, cap)
            
            decode_score, decode_pre = torch.max(decode_out, dim = -1)
            
            caps = tokenizer.token2text(cap)
            predicts = tokenizer.token2text(decode_pre)
            
            bleu_score_his = corpus_bleu(caps, predicts)
            total_test_bleu += bleu_score_his
        return bleu_score_his/len(data_loader)

def testing_in_a_img(img_path, encoder, decoder, tokenizer, device, visual = False):
    '''Get the caption of a specific image'''
    img = Image.open(img_path)
    
    image_features = vgg16(img)
    batch, channel, _, _ = image_features.shape
    image_features = image_features.view(batch, channel, -1)
    features = torch.unsqueeze(image_features, dim = 0)
    features = encoder(features)
    
    out, hn, attention = decoder(features)
    
    scores, predicts = torch.max(out, dim = -1)
    
    predicts_text = tokenizer.token2text(predicts)
    
    if visual:
        visual_attentions(predicts_text, attention)
    
    text_predict = ""
    for text in predicts:
        if text == "<end>":
            break
        text_predict += text
    return text_predict
        