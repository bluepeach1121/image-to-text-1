import os
import json
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet34, ResNet34_Weights


# ------------------------------------------------------------------------
# 1) COCO-like Dataset for val2017
# ------------------------------------------------------------------------

class CocoValDataset(Dataset):
    """
    A minimal COCO-like dataset class for val2017 images & captions.
    We'll do a small train/test split on the val set for demonstration.
    """
    def __init__(
        self, 
        image_dir, 
        annotation_file, 
        image_ids, 
        transform=None, 
        max_caption_len=20, 
        word2idx=None, 
        idx2word=None, 
        sos_token='<SOS>', 
        eos_token='<EOS>'
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.max_caption_len = max_caption_len
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        with open(annotation_file, 'r') as f:
            coco_anns = json.load(f)
        
        # Build a mapping {image_id -> [list_of_captions]}
        self.img_id_to_caps = {}
        for ann in coco_anns['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_caps:
                self.img_id_to_caps[img_id] = []
            self.img_id_to_caps[img_id].append(ann['caption'])
        
        # Build a mapping {img_id -> file_name} from images
        self.img_id_to_file = {}
        for img_info in coco_anns['images']:
            self.img_id_to_file[img_info['id']] = img_info['file_name']
        
        # Subset of image IDs (train or test split)
        self.image_ids = image_ids
        
        # Pre-computed vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        file_name = self.img_id_to_file[img_id]
        path = os.path.join(self.image_dir, file_name)
        
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        captions = self.img_id_to_caps[img_id]
        caption_str = random.choice(captions)

        # Convert caption to indices
        tokens = caption_str.lower().strip().split()
        tokens = [self.sos_token] + tokens + [self.eos_token]
        
        token_ids = []
        for w in tokens:
            token_ids.append(self.word2idx.get(w, self.word2idx['<UNK>']))
        
        token_ids = token_ids[:self.max_caption_len]
        token_ids += [self.word2idx['<PAD>']] * (self.max_caption_len - len(token_ids))
        
        return image, torch.tensor(token_ids)

# ------------------------------------------------------------------------
# 2) Vocab Builder
# ------------------------------------------------------------------------

def build_vocab(annotation_file, min_freq=2):
    """
    function to build a vocabulary from all captions in COCO val2017.
    Words appearing less frequently are excluded.

    """
    with open(annotation_file, 'r') as f:
        coco_anns = json.load(f)
    
    word_freq = {}
    for ann in coco_anns['annotations']:
        caption = ann['caption']
        for w in caption.lower().strip().split():
            word_freq[w] = word_freq.get(w, 0) + 1
    
    # Keep words above min_freq
    words = [w for w, c in word_freq.items() if c >= min_freq]
    
    # Special tokens
    idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + sorted(words)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    
    return word2idx, idx2word
