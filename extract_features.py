from datasets import load_data_from_path
from config import *
from PIL import Image 
from model.vgg16_base import VGG16
from torchvision import transforms
import torch
import os.path as osp
import os
from tqdm import tqdm
images_path, data = load_data_from_path(image_path, annotation_path)
vgg = VGG16()
def extrat(path):
    transform = transforms.Compose([
        transforms.Resize(size = (224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(path)
    img = transform(img)
    img = torch.unsqueeze(img, dim = 0)
    img = vgg(img)
    img = img.view(1, -1, 256)
    
    return img

def extrat_features(data, root_path):
    if not osp.exists(path=root_path):
        os.makedirs(root_path)
    for path in tqdm(data):
        img_path = osp.join(image_path, path)
        
        feature = extrat(img_path)
        
        feature_path_save = osp.join(root_path, path).replace(".jpg", ".pt")
        
        torch.save(feature, feature_path_save)

extrat_features(images_path, path_feature)