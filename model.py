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
# 3) Encoder (ResNet-34) + Decoder (LSTM)
# ------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(self, embed_size=256):
        super(CNNEncoder, self).__init__()
        # Use a pretrained ResNet-34
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        # Remove the final FC layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Project to embed_size
        self.linear = nn.Linear(512, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 512]
        features = self.linear(features)                # [B, embed_size]
        features = self.bn(features)                    # [B, embed_size]
        return features

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        """
        features: [B, embed_size]
        captions: [B, max_len]
        teacher_forcing_ratio: probability to use ground-truth token each step
        """
        batch_size, max_len = captions.shape
        
        # Expand features from [B, embed_size] -> [B, 1, embed_size]
        features = features.unsqueeze(1)
        
        # Pre-embed the entire caption
        embeddings = self.embed(captions)  # [B, max_len, embed_size]
        
        # Initialize hidden state
        h, c = self.init_hidden_state(batch_size, device=features.device)
        
        outputs = []
        current_input = embeddings[:, 0, :]  # <SOS> embedding
        for t in range(max_len - 1):
            lstm_input = current_input.unsqueeze(1)
            
            out, (h, c) = self.lstm(lstm_input, (h, c))   # out: [B, 1, hidden_size]
            out = self.fc(out.squeeze(1))                 # [B, vocab_size]
            
            outputs.append(out.unsqueeze(1))
            
            # Decide next input (teacher forcing vs. model’s own prediction)
            use_teacher_forcing = (random.random() < teacher_forcing_ratio)
            if use_teacher_forcing and t < (max_len - 1):
                current_input = embeddings[:, t+1, :]
            else:
                _, predicted = out.max(dim=1)
                current_input = self.embed(predicted)
                
        outputs = torch.cat(outputs, dim=1)  # [B, max_len-1, vocab_size]
        return outputs
    
    def init_hidden_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h, c
