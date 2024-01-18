from typing import Any
from matplotlib import pyplot as plt 
from keras.preprocessing.image import load_img
import textwrap
import string
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
import torchdata.datapipes as dp
import torchtext.transforms as T
import numpy as np
def visualizing(uni_filenames, images_root_path, data):
    npic = len(uni_filenames)
    npix = 224
    target_size = (npix,npix,3)
    count = 1

    fig = plt.figure(figsize=(12,7))
    for jpgfnm in uni_filenames:
        filename = images_root_path + jpgfnm
        captions = list(data["caption"].loc[data["filename"]==jpgfnm].values)
        image_load = load_img(filename, target_size=target_size)
        ax = fig.add_subplot(npic,3,count,xticks=[],yticks=[])
        ax.imshow(image_load)
        count += 1

        ax = fig.add_subplot(npic,3,count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,2)
        ax.set_ylim(0,len(captions))
        for i, caption in enumerate(captions):
            ax.text(0, i, textwrap.fill(caption, width=100), fontsize=11, wrap=True)
        count += 2
    plt.show()
def remove_single_charater(caption):
    text_result = ""
    for text in caption.split():
        if(len(text)) > 1:
            text_result += " "+text
    return text_result
def remove_numric(text):
    text_no_numberic = ""
    for txt in text.split():
        if txt.isalpha():
            text_no_numberic += " " + txt
    return text_no_numberic
def remove_punctuation(text):
    translation_table = str.maketrans("", "", string.punctuation)
    text_no_punctuation = text.translate(translation_table)
    return (text_no_punctuation)

def text_clean(caption):
    text = remove_punctuation(caption)
    text = remove_numric(text)
    text = remove_single_charater(text)
    return text.lower()
def add_tags_start_end(captions):
    text = "" + captions + ""
    return text.replace("  ", " ")

# Build vocabulary
def get_token(data):
        for txt in data:
            yield txt.split()
            
def get_vocabulary(captions):
    source_vocab = build_vocab_from_iterator(
    get_token(captions),
    min_freq=2,
    special_first=True,
    specials=['<pad>', '<start>', '<end>', '<unk>']  # Corrected here
    )
    source_vocab.set_default_index(source_vocab['<unk>'])
    return source_vocab
class Tokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.source_vocab = vocab.get_itos()
        
    def get_transform(self):
        """
        Create transforms based on given vocabulary. The returned transform is applied to sequence of tokens
        """
        text_transform = T.Sequential(
            T.VocabTransform(vocab= self.vocab),
            T.AddToken(1, begin= True),
            T.AddToken(2, begin=False)
        )
        return text_transform
    
    def __call__(self, caption):
        return self.encode(caption)
    
    def encode(self, caption):
        some_sentences = caption.split()
        transform = self.get_transform()
        return transform(some_sentences)
    
    def decode(self, token):
        caption =  ""
        for index in token:
            caption += self.source_vocab[index] + " "
        return caption.strip()
    
    def text2token(self, captions, padding = True):
        captions = list(map(self.encode, captions))
        if padding:
            captions = T.ToTensor(0)(captions)
        return captions
    
    def token2text(self, tokens):
        tokens = list(map(self.decode, tokens))
        tokens = np.array(tokens)
        return tokens