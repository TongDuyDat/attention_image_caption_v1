from sklearn.model_selection import train_test_split
from utils import Tokenizer
from torchvision import transforms
from model.vgg16_base import VGG16
from torch.utils.data import DataLoader
from datasets import load_data_from_path, data_after_process, ImageCaptionDataset
from config  import *
import pandas as pd
from utils import get_vocabulary
import numpy as np
#conts               
pd.options.mode.chained_assignment = None

def data_train_test_split(data, root_path_text = None):
    # Split the dataset into training and test set using 75% of the data for training
    # and 25% of the data for testing. The random state is fixed to ensure
    # reproducibility.
    if root_path_text is None:
        image_paths, captions = data_after_process(data, root_path= image_path)
        vocabulary = get_vocabulary(captions)
        X_train, X_test, y_train, y_test = train_test_split(image_paths, captions, test_size = 0.2)
        return {
            "train": (X_train, y_train), 
            "val": (X_test, y_test)
        }, vocabulary
        
    train_path = root_path_text+"/Flickr_8k.trainImages.txt"
    val_path = root_path_text+"/Flickr_8k.devImages.txt"
    test_path = root_path_text+"/Flickr_8k.testImages.txt"
    
    with open(train_path, "r") as file:
        train_data_set = file.read().splitlines()
        df_train = pd.DataFrame(train_data_set, columns = ["filename"])
        mask = data["filename"].isin(df_train["filename"])
        df_train = data[mask]
        
    with open(val_path, "r") as file:
        val_data_set = file.read().splitlines()
        df_val = pd.DataFrame(val_data_set, columns = ["filename"])
        mask = data["filename"].isin(df_val["filename"])
        df_val = data[mask]
        
    with open(test_path, "r") as file:
        test_data_set = file.read().splitlines()
        df_test = pd.DataFrame(test_data_set, columns = ["filename"])
        mask = data["filename"].isin(df_test["filename"])
        df_test = data[mask]
    # image_paths, captions = data_after_process(data, image_path)
    
    image_paths_train, captions_train = data_after_process(df_train, image_path)
    image_paths_val, captions_val = data_after_process(df_val, image_path)
    image_paths_test, captions_test = data_after_process(df_test, image_path)
    all_caption = np.concatenate([captions_train, captions_val, captions_test], axis= 0)
    vocabulary = get_vocabulary(all_caption)
    
    return {
        "train": (image_paths_train, captions_train),
        "val": ( image_paths_val, captions_val),
        "test": (image_paths_test, captions_test)
    }, vocabulary

# def data_train_test_val(data, batch, phase = 'train'):
#     images, captions = data["train"]
#     dataset = ImageCaptionDataset(images, captions, transform)
#     dataloader_train = DataLoader(dataset, batch_size=batch, suffle=False)
    
#     images, captions = data["val"]
#     dataset = ImageCaptionDataset(images, captions, transform)
#     dataloader_val = DataLoader(dataset, batch_size=batch, suffle=False)
    
#     images, captions = data["val"]
#     dataset = ImageCaptionDataset(images, captions, transform)
#     dataloader_test = DataLoader(dataset, batch_size=batch, suffle=False)
    
#     return dataloader_train, dataloader_val, dataloader_test
def data_loader(image_path, annotation_path, batch_size):
    """
        Load the image and the corresponding annotation from a folder containing image files and an annotation file.
        Load the Coco-Captions dataset from disk and initialize a torch Dataset object.
        The returned object can be used in a DataLoader for easy access to the individual elements of the
        dataset during training.
        Args:
        root: path to the directory where the files are located. Default is './data'.
        transform: an instance of class that inherits from Transform. Default is None. See below
        for details on how to create your own custom transformation classes.
        Returns:
        A Pytorch Dataset object.
        Example usage:
    """
    
    
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])
    
    
    print("Loading data in local .............")
    _, data = load_data_from_path(image_path, annotation_path)
    data, vocab = data_train_test_split(data, root_path_text)
    
    # get max length
    max_length = 0
    for image, caption in data.values():
        max_length = max(max_length, len(max(caption, key = len).split()))
        
    tokenizer = Tokenizer(vocab, max_length)
    
    images_train, caption_train = data["train"]
    images_val, caption_val = data["val"]
    images_test, caption_test = data["test"]
    
    train_captions = tokenizer.text2token(caption_train)
    val_captions = tokenizer.text2token(caption_val) 
    test_captions = tokenizer.text2token(caption_test) 
    
    dataset_train = ImageCaptionDataset(images_train, train_captions, transform)
    dataset_val = ImageCaptionDataset(images_val, val_captions, transform)
    dataset_test = ImageCaptionDataset(images_test, test_captions, transform)
    
    print("Shape train", len(data["train"][0]), len(data["train"][1]))
    print("Shape val", len(data["val"][0]), len(data["val"][1]))
    print("Shape test", len(data["test"][0]), len(data["test"][1]))
    print("Vocabluary: ", len(vocab))
    
    train_loader = DataLoader(dataset_train, batch_size, shuffle= True)
    val_loader = DataLoader(dataset_val, batch_size, shuffle= True)
    test_loader = DataLoader(dataset_test, batch_size, shuffle= True)
    return train_loader, val_loader, test_loader, tokenizer