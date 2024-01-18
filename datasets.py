import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.utils import shuffle 
from PIL import Image
# from transformers import AutoTokenizer
from utils import Tokenizer
from utils import visualizing, text_clean, add_tags_start_end, get_vocabulary
import torch
from torchvision import transforms
from model.vgg16_base import VGG16
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

#function 
def load_data_from_path(image_path, annotation_path):
    images = glob(image_path + "/*")
    print(len(images))

    with open(annotation_path, "r") as file:
        annotations = [line.strip().split("\t") for line in file]
        data = []
        for context in annotations:
            if len(context)==1:
                continue
            data.append(context[0].split("#")+ [context[1].lower()])
    data = pd.DataFrame(data, columns = ["filename", "index", "caption"])
    data = data.reindex(columns= ["index", "filename", "caption"])
    data = data[data.filename != '2258277193_586949ec62.jpg.1']
    uni_filenames = np.unique(data.filename.values)
    return uni_filenames, data

# visualizing(uni_filenames[10:14], image_path, data)

def data_after_process(data, root_path, num_limted = None):
    data_new = data
    for i, (filename, caption) in enumerate(zip(data_new.filename.values, data_new.caption.values)):
        new_caption = text_clean(caption)
        new_caption = add_tags_start_end(new_caption)
        data_new["caption"].iloc[i] = new_caption
        data_new["filename"].iloc[i] = root_path + "/" + filename
    image_paths = data_new.filename.values
    captions = data_new.caption.values
    vocabulary = get_vocabulary(captions)
    
    if num_limted is not None:
        image_paths, captions = shuffle(image_paths, captions, random_state= 1)
        image_paths = image_paths[: num_limted]
        captions = captions[: num_limted]
    return image_paths, captions, vocabulary


class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, transform):
        self.image_paths = image_paths
        self.captions = captions
        # self.transform_img = transform_img
        self.transform_caption = transform_caption
        model = VGG16()
        self.model = model.eval()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        # preprocessing the image
        image = Image.open(img_path)
        image = self.transform_img(image)
        image = torch.unsqueeze(image, dim = 0)
        imgae_features = self.model(image)
        batch, channel, _, _ = imgae_features.shape
        imgae_features = imgae_features.view(batch, channel, -1)
        
        # processing caption 
        
        # caption = transform_caption(caption)
        
        return imgae_features, caption
            
# const 

#function get features
# def images_features(image_paths):
#     features = []
#     for image in tqdm(image_paths):
#         img = Image.open(image).convert("RGB")
#         img = transform(img= img)
#         img = torch.unsqueeze(img, dim = 0)
#         feature = vgg16(img)
#         feature = feature.view(feature.shape[0], feature.shape[1], -1)
#         features.append(feature)
#         del img
#     return features

# def captions_features(captions, tokenizer):
#     outs = tokenizer.text2token(captions)
#     return outs
        

# img_features = images_features(image_paths)
# caption_features = captions_features(captions)

# print(len(img_features))
# print(len(caption_features))