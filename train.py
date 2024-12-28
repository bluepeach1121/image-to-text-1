import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # <- ADD THIS IMPORT


def train_one_epoch(
    encoder, 
    decoder, 
    dataloader, 
    criterion, 
    encoder_optimizer,
    decoder_optimizer, 
    device, 
    epoch, 
    total_epochs, 
    initial_teacher_forcing=1.0, 
    final_teacher_forcing=0.5
):
    """
    Trains for one epoch, with a linear schedule for teacher_forcing_ratio.
    Now includes a tqdm progress bar.
    """
    encoder.train()
    decoder.train()
    
    # Linear schedule for teacher_forcing_ratio
    teacher_forcing_ratio = (
        (initial_teacher_forcing - final_teacher_forcing) 
        * (1 - epoch / total_epochs) 
        + final_teacher_forcing
    )
    
    total_loss = 0.0
    
    # Wrap the dataloader with tqdm for a progress bar
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [TRAIN]")
    
    for images, captions in loop:
        images = images.to(device)
        captions = captions.to(device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        features = encoder(images)
        outputs = decoder(features, captions, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # outputs: [B, max_len-1, vocab_size]
        targets = captions[:, 1:]  # shift ground truth
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        total_loss += loss.item()
        
        # Update tqdm with the current loss
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)


def evaluate_model(encoder, decoder, dataloader, device):
    """
    Minimal evaluation (average cross-entropy).
    Includes a tqdm bar for progress.
    """
    encoder.eval()
    decoder.eval()
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # assuming <PAD> = 0
    total_loss = 0.0
    
    # Wrap the dataloader with tqdm for a progress bar
    loop = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for images, captions in loop:
            images = images.to(device)
            captions = captions.to(device)
            
            features = encoder(images)
            outputs = decoder(features, captions, teacher_forcing_ratio=0.0)
            
            targets = captions[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            
            # Update tqdm with the current batch's loss
            loop.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)



