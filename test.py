import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from config import *
from dataloder import data_loader
from model.vgg16_base import VGG16
from PIL import Image
from utils import visual_attentions, load_model
from torchvision import transforms
from model.attention import Encode, Decode
trainloader, valloader, testloader, tokenizer = data_loader(image_path, annotation_path, batch_size= 32)
vgg16 = VGG16()
def testing(data_loader, model, tokenizer, device):
    '''Testing phase'''
    model.eval()
    
    total_test_bleu = 0.0
    
    with torch.no_grad():
        for img, cap in tqdm(data_loader):
            img = img.to(device)
            cap = cap.to(device)
            encoder_out = model.encode(img)
            decode_out, decode_hidden, attetions = model.decode(encoder_out, cap, 0)
            
            decode_score, decode_pre = torch.max(decode_out, dim = -1)
            
            caps = tokenizer.token2text(cap)
            predicts = tokenizer.token2text(decode_pre)
            print(" ".join(predicts[0]))
            bleu_score_his = corpus_bleu(caps, predicts, smoothing_function= SmoothingFunction().method1)
            total_test_bleu += bleu_score_his
        return bleu_score_his/len(data_loader)

def testing_in_a_img(img_path, encoder, decoder, tokenizer, device, visual = False):
    '''Get the caption of a specific image'''
    encoder.eval()
    decoder.eval()
    img = Image.open(img_path)
    img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])(img)
    img = torch.unsqueeze(img, dim = 0)
    image_features = vgg16(img)
    batch, channel, _, _ = image_features.shape
    image_features = image_features.view(-1, channel)
    features = torch.unsqueeze(image_features, dim = 0)
    features = encoder(features)
    
    out, hn, attention = decoder(features)
    
    out = torch.nn.functional.softmax(out, dim = -1)
    scores, predicts = torch.max(out, dim = -1)
    print(attention.shape)
    predicts_text = tokenizer.token2text(predicts)
    
    if visual:
        visual_attentions(predicts_text, attention)
    text_predict = ""
    for text in predicts:
        print(text)
        if text == "<end>":
            break
        text_predict += text
    return text_predict
        
        
# encode = Encode(embedding_dim, embedding_dim)
# decode = Decode(embedding_dim, units, 5120)

# encode = load_model(encode, "train/model/best_encode.pt")
# decode = load_model(decode, "train/model/best_decode.pt")

# img_pathsss = "D:/NCKH/ImageCaption/Dataset/Flickr8k_Dataset/Flicker8k_Dataset/10815824_2997e03d76.jpg"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# s = testing_in_a_img(img_pathsss, encode, decode, tokenizer, device, visual= True)