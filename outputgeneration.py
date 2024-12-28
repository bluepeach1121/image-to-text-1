# outputgeneration.py

import torch
from PIL import Image

def generate_caption(
    encoder, 
    decoder, 
    image_path, 
    transform, 
    device, 
    word2idx, 
    idx2word, 
    max_length=20
):
    """
    1) Loads and preprocesses the image.
    2) Feeds it to the encoder -> features [1, embed_size].
    3) Greedy decodes via the LSTM, step by step.
    4) Returns the generated caption string.

    Includes debug prints to show tensor shapes at each step.
    """
    # ---------------------------------------------------------------------
    # Special tokens
    # ---------------------------------------------------------------------
    start_token = word2idx['<SOS>']
    end_token   = word2idx['<EOS>']
    
    # ---------------------------------------------------------------------
    # 1) Load & Preprocess Image
    # ---------------------------------------------------------------------
    image = Image.open(image_path).convert('RGB')
    image = transform(image)               # e.g. [3, 224, 224]
    image = image.unsqueeze(0).to(device)  # [1, 3, 224, 224]
    
    # ---------------------------------------------------------------------
    # 2) Encode
    # ---------------------------------------------------------------------
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        features = encoder(image)  # Expect shape [1, embed_size]
        print("DEBUG: encoder output shape =", features.shape)
        # Example: torch.Size([1, 256])

    # ---------------------------------------------------------------------
    # 3) Greedy Decode (One Word at a Time)
    # ---------------------------------------------------------------------
    # We'll manually pass each predicted token into the LSTM.

    # Initialize hidden/cell states
    batch_size = 1
    hidden_state, cell_state = decoder.init_hidden_state(batch_size, device=device)

    # Current input token = <SOS>
    # shape [1], i.e. just one token, for batch size 1
    current_input = torch.tensor([start_token], device=device)  # shape [1]
    # We'll expand it to [1,1] so it becomes [batch=1, seq_len=1]
    current_input = current_input.unsqueeze(0)  # shape [1,1]

    print("DEBUG: initial current_input shape =", current_input.shape)
    # Should be [1,1], meaning batch=1, seq_len=1

    generated_indices = []

    for step in range(max_length):
        # -----------------------------------------------------------------
        # Embed the current token
        # -----------------------------------------------------------------
        word_embed = decoder.embed(current_input)  # shape [1,1,embed_size]
        print(f"DEBUG: Step {step}: word_embed shape =", word_embed.shape)

        # LSTM step
        out, (hidden_state, cell_state) = decoder.lstm(word_embed, (hidden_state, cell_state))
        # out shape: [1,1,hidden_size] if batch_first=True

        # Project to vocabulary dimension
        out = decoder.fc(out.squeeze(1))  # shape [1, vocab_size]
        
        # Pick the highest-probability token
        _, predicted = out.max(dim=1)     # shape [1]
        predicted_idx = predicted.item()
        generated_indices.append(predicted_idx)
        
        # If we hit <EOS>, stop
        if predicted_idx == end_token:
            break
        
        # Prepare input for next step
        # predicted is shape [1], so unsqueeze(0) -> [1,1]
        current_input = predicted.unsqueeze(0)

    # ---------------------------------------------------------------------
    # 4) Convert Word Indices to Actual Tokens
    # ---------------------------------------------------------------------
    caption_tokens = []
    for idx in generated_indices:
        word = idx2word[idx]
        if word == '<EOS>':
            break
        if word not in ('<SOS>', '<PAD>'):
            caption_tokens.append(word)
    
    caption_str = " ".join(caption_tokens)
    return caption_str
