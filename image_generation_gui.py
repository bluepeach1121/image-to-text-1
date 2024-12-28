import torch
import pickle
from tkinter import Tk, filedialog
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import CNNEncoder, LSTMDecoder
from outputgeneration import generate_caption


def main():
    #Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    #Load vocabulary
    with open("vocab.pkl", "rb") as f:
        word2idx, idx2word = pickle.load(f)
    vocab_size = len(word2idx)
    print("Vocabulary loaded. Vocab size:", vocab_size)

    embed_size = 256
    hidden_size = 512
    encoder = CNNEncoder(embed_size=embed_size).to(device)
    decoder = LSTMDecoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size).to(device)

    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
    encoder.eval()
    decoder.eval()
    print("Models loaded successfully.")

    #image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #Use tkinter to pick an image
    print("Opening file dialog to select an image...")
    root = Tk()
    root.withdraw()  
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    
    if not image_path:
        print("No file selected. Exiting.")
        return
    
    print(f"Image selected: {image_path}")

    #Load and preprocess the image
    image = Image.open(image_path).convert('RGB')

    #Generate caption
    caption = generate_caption(
        encoder=encoder,
        decoder=decoder,
        image_path=image_path,
        transform=transform,
        device=device,
        word2idx=word2idx,
        idx2word=idx2word,
        max_length=20
    )
    print("Generated Caption:", caption)

    #Display the image with the caption
    plt.imshow(image)
    plt.title(caption, fontsize=12)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
