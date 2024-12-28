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


# Our local imports
from caption_dataset import CocoValDataset, build_vocab
from model import CNNEncoder, LSTMDecoder
# from train.py, which now includes tqdm
from train import train_one_epoch, evaluate_model


def main():
    ANNOTATION_FILE = r"C:\Users\olawa\Downloads\annotations_trainval2017\annotations\captions_val2017.json"
    IMAGE_DIR = r"C:\Users\olawa\Downloads\val2017\val2017"
    
    #Build vocab
    print("Building vocabulary...")
    word2idx, idx2word = build_vocab(ANNOTATION_FILE, min_freq=2)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    
    #Load image IDs
    with open(ANNOTATION_FILE, 'r') as f:
        coco_anns = json.load(f)
    all_img_ids = [img['id'] for img in coco_anns['images']]
    
    # Train/test split
    train_ids, test_ids = train_test_split(all_img_ids, test_size=0.2, random_state=42)
    print(f"Train set size: {len(train_ids)} | Test set size: {len(test_ids)}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Create Datasets
    train_dataset = CocoValDataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        image_ids=train_ids,
        transform=transform,
        word2idx=word2idx,
        idx2word=idx2word
    )
    test_dataset = CocoValDataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        image_ids=test_ids,
        transform=transform,
        word2idx=word2idx,
        idx2word=idx2word
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    #Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size = 256
    hidden_size = 512
    
    encoder = CNNEncoder(embed_size=embed_size).to(device)
    decoder = LSTMDecoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=1e-4)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=1e-4)
    
    #Training loop with tqdm
    epochs = 6
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            encoder,
            decoder,
            train_loader,
            criterion,
            encoder_optimizer,
            decoder_optimizer,
            device,
            epoch=epoch,
            total_epochs=epochs,
            initial_teacher_forcing=1.0,
            final_teacher_forcing=0.5
        )
        
        val_loss = evaluate_model(encoder, decoder, test_loader, device)
        
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print("Training complete!, saving model weights")
    print("Training complete! Saving model...")

    # Save encoder
    torch.save(encoder.state_dict(), "encoder.pth")

    # Save decoder
    torch.save(decoder.state_dict(), "decoder.pth")

    #Save word2idx, idx2word --> I dont think I'll need this but whatever
    import pickle
    with open("vocab.pkl", "wb") as f:
        pickle.dump((word2idx, idx2word), f)

    print("Models and vocab saved.")

if __name__ == "__main__":
    main()
