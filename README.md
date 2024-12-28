# Image-to-Text Model 1 (First attempt)

This project implements an **image-to-text system** that generates descriptive captions for images using a **CNN encoder (resnet-34)** and an **LSTM decoder**. The system is trained on the val2017 COCO dataset and can generate captions for new images.

---

## **Project Structure**

### **Files and Directories**
- `caption_dataset.py`: Contains the `CocoValDataset` class for handling COCO dataset loading and preprocessing.
- `model.py`: Defines the `CNNEncoder` and `LSTMDecoder` classes.
- `train.py`: Contains functions for training and evaluating.
- `outputgeneration.py`: Implements the `generate_caption` function for inference (generating captions for new images).
- `usage1.py`: Example script to test the model on a single image and generate captions.
- `usage2.py`: Another usage script.
- `main.py`: The main script to train the model.
# Limitations of the Image-to-Text Captioning Model

This project implements a basic image captioning model using a **CNN encoder** and an **LSTM decoder**. While functional, the model has several limitations:

## Key Limitations

### 1. Overfitting
- **Cause**: Small training dataset (COCO val2017 with 5,000 images).
- **Fix**: Train on COCO train2017 (118k images) and use regularization (dropout, weight decay).

### 2. Small CNN Encoder
- **Cause**: ResNet-34 used as the encoder.
- **Fix**: Use larger encoders (ResNet-50, EfficientNet, or Vision Transformers).

### 3. LSTM Decoder
- **Cause**: LSTM's sequential nature and limited long-term dependency handling.
- **Fix**: Replace LSTM with a Transformer-based decoder.

### 4. Greedy Decoding
- **Cause**: Always picks the most likely word without exploring alternatives.
- **Impact**: Repetitive or incomplete captions.
- **Fix**: Implement beam search or top-k sampling for better decoding.

### 5. Small Training Dataset
- **Cause**: Training on val2017 instead of train2017.
- **Impact**: Poor generalization to unseen images.
- **Fix**: Use larger datasets like COCO train2017 or Flickr30k.

### 6. Lack of Metrics
- **Cause**: No BLEU, CIDEr, or ROUGE metrics implemented.

### 7. Limited Vocabulary
- **Cause**: Vocabulary restricted to the small training set.
- **Fix**: Use pretrained embeddings (e.g., GloVe, Word2Vec).

---

## Summary

| Limitation            | Cause                   | Fix                                |
|-----------------------|-------------------------|------------------------------------|
| Overfitting           | Small dataset          | Train on larger datasets          |
| Small CNN Encoder     | ResNet-34              | Use ResNet-50, ViT, or EfficientNet |
| LSTM Decoder          | Sequential model       | Replace with Transformer Decoder  |
| Greedy Decoding       | No exploration         | Use Beam Search                   |
| Small Training Dataset| val2017 subset         | Use train2017                     |
| Lack of Metrics       | No BLEU, CIDEr, etc.   | Add quantitative evaluation       |
| Limited Vocabulary    | Small vocab size       | Use pretrained embeddings         |

---

This README highlights the limitations of the model and suggests improvements for better performance.
