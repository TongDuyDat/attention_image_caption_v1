import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from config import *
from dataloder import data_loader
from model.vgg16_base import VGG16
from PIL import Image
from utils import visual_attentions, load_model
from torchvision import transforms
from model.attention import Encode, Decode, Squen2Squen
# trainloader, valloader, testloader, tokenizer = data_loader(image_path, annotation_path, batch_size= 32)
vgg16 = VGG16()
def testing(data_loader, model, tokenizer, device):
    '''Testing phase'''
    model.eval()
    
    total_test_bleu = 0.0
    
    with torch.no_grad():
        for img, cap in tqdm(data_loader):
            img = img.to(device)
            cap = cap.to(device)
            decode_out, decode_hidden, attetions = model(img, cap, 0)
            
            decode_score, decode_pre = torch.max(decode_out, dim = -1)
            
            caps = tokenizer.token2text(cap)
            predicts = tokenizer.token2text(decode_pre)
            bleu_score_his = corpus_bleu(caps, predicts, smoothing_function= SmoothingFunction().method1)
            total_test_bleu += bleu_score_his
        return bleu_score_his/len(data_loader)

def testing_in_a_img(img_path, model, tokenizer, device, visual = False):
    '''Get the caption of a specific image'''

    model.eval()
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
    features = model.encode(features)
    
    text = []
    attens = []
    hidden = model.decode.reset_state(1)
    x = torch.ones(size=(1, 1), dtype=torch.long)
    for i in range(tokenizer.max_length):
        x, hidden, atten = model.decode(x, features, hidden)
        x = torch.nn.functional.softmax(x, dim = -1)
        scores, predicts = torch.max(x, dim = -1)
        print(predicts)
        x = torch.unsqueeze(predicts, dim =0)
        predicts_text = tokenizer.decode(predicts)
        text.append(predicts_text)
        attens.append(atten)
        if predicts_text == "<end>":
            break
    if visual:
        print(attens)
        attens = torch.cat(attens, dim=0)
        visual_attentions(text, attens)
    
    return " ".join(text)
        
        
# encode = Encode(embedding_dim, embedding_dim)
# decode = Decode(embedding_dim, units, 5120)

# encode = load_model(encode, "train/model/best_encode.pt")
# decode = load_model(decode, "train/model/best_decode.pt")

# model = Squen2Squen(encode, decode, tokenizer.max_length)

# img_pathsss = "D:/NCKH/ImageCaption/Dataset/Flickr8k_Dataset/Flicker8k_Dataset/10815824_2997e03d76.jpg"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# s = testing_in_a_img(img_pathsss, model, tokenizer, device, visual= True)
# print(s)