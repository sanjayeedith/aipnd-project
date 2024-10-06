import torch
from PIL import Image
import argparse
import json
import numpy as np
from torchvision import models

# Parsing command line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model')
    
    # Positional arguments
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the trained model checkpoint')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

# Function to load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Function to process an image
def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    np_image = np.array(image) / 255
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(np_image.transpose((2, 0, 1)))

# Function to predict the class of an image
def predict(image_path, model, topk=1, device='cpu'):
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = image.unsqueeze(0).float()
    image = image.to(device)
    
    with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps)
        top_ps, top_classes = ps.topk(topk, dim=1)
    
    return top_ps.cpu().numpy(), top_classes.cpu().numpy()

# Main function
if __name__ == "__main__":
    args = get_input_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_checkpoint(args.checkpoint)
    
    # Predict
    probs, classes = predict(args.input, model, args.top_k, device)
    
    # If category_names is provided, map the classes to names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes[0]]
    else:
        class_names = classes[0]
    
    print(f"Predicted Classes: {class_names}")
    print(f"Probabilities: {probs[0]}")
