import torch
import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
import os

# Parsing command line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Train a neural network for flower classification')
    
    # Positional argument for data directory
    parser.add_argument('data_dir', type=str, help='Directory containing training and validation data')
    
    # Optional arguments for model hyperparameters and architecture
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the trained model checkpoint')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture (e.g., "vgg13", "vgg16")')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()

# Function to train the model
def train_model(args):
    # Define transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }
    
    # Load datasets
    train_data = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), transform=data_transforms['valid'])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    # Load pre-trained model
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define the classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop (simplified for demonstration)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}.. Training loss: {running_loss/len(train_loader):.3f}")
    
    # Save checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': args.arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))

if __name__ == "__main__":
    args = get_input_args()
    train_model(args)
