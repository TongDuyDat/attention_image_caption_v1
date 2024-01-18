from sklearn.model_selection import train_test_split
from utils import Tokenizer
from torchvision import transforms
from model.vgg16_base import VGG16
from torch.utils.data import DataLoader
from datasets import load_data_from_path, data_after_process, ImageCaptionDataset

#conts
image_path = "D:/NCKH/ImageCaption/Dataset/Flickr8k_Dataset/Flicker8k_Dataset"
annotation_path = "D:/NCKH/ImageCaption/Dataset/Flickr8k_text/Flickr8k.token.txt"  
root_path_text =  "D:/NCKH/ImageCaption/Dataset/Flickr8k_text"               
uni_filenames, data = load_data_from_path(image_path, annotation_path)
image_paths, captions, vocabulary = data_after_process(data, image_path, num_limted=40000)
print("Number of Images : ", len(image_paths), "\nNumber of Captions: ", len(captions), "\nNumber of Vocabulary: ", len(vocabulary))

model_base = VGG16()
tokenizer = Tokenizer(vocabulary)
transform = transforms.Compose([
    transforms.Resize(size = (224, 224)), 
    transforms.ToTensor(),
])
def data_train_test_split(image_path, caption, root_path_text):
    # Split the dataset into training and test set using 75% of the data for training
    # and 25% of the data for testing. The random state is fixed to ensure
    # reproducibility.
    train_path = root_path_text+"/Flickr_8k.trainImages.txt"
    val_path = root_path_text+"/Flickr_8k.devImages.txt"
    test_path = root_path_text+"/Flickr_8k.testImages.txt"
    
    train_caption, val_caption, test_caption = [], [], []
    train_img, val_img, test_img = [], [], []
    train_data_set = []
    with open(train_path, "r") as file:
        train_data_set = file.read().splitlines()
        
    val_data_set = []
    with open(val_path, "r") as file:
        val_data_set = file.read().splitlines()
    
    test_data_set = []
    with open(test_path, "r") as file:
        test_data_set = file.read().splitlines()
    
    for image, caption in zip(image_path, caption):
        if image.split("/")[-1] in train_data_set:
            train_caption.append(caption)
            train_img.append(image)
        elif image.split("/")[-1] in val_data_set:
            val_caption.append(caption)
            val_img.append(image)
        elif image.split("/")[-1] in test_data_set:
            test_caption.append(caption)
            test_img.append(image)
    return {
        "train": (train_img, train_caption),
        "val": (val_img, val_caption),
        "test": (test_img, test_caption)
    }

def data_train_test_val(batch, phase = 'train'):
    data  = data_train_test_split(image_path, captions, root_path_text)
    images, captions = data["train"]
    dataset = ImageCaptionDataset(images, captions, model_base, transform)
    dataloader_train = DataLoader(dataset, batch_size=batch, suffle=False)
    
    images, captions = data["val"]
    dataset = ImageCaptionDataset(images, captions, model_base, transform)
    dataloader_val = DataLoader(dataset, batch_size=batch, suffle=False)
    
    images, captions = data["val"]
    dataset = ImageCaptionDataset(images, captions, model_base, transform)
    dataloader_test = DataLoader(dataset, batch_size=batch, suffle=False)
    
    return dataloader_train, dataloader_val, dataloader_test
       