# src/train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import MedicalImageModel
from dataset import get_dataloaders
import os

def train_model(data_dir, num_epochs=5):
    # 1. Setup device (Apple Silicon MPS, Nvidia CUDA, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load Data
    dataloaders, class_names = get_dataloaders(data_dir)
    print(f"Classes found: {class_names}")

    # 3. Initialize Model, Loss, and Optimizer
    model = MedicalImageModel(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.base_model.fc.parameters(), lr=0.001) # Only training the final layer for now

    # 4. Training Loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save the trained model
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/medical_cnn_weights.pth')
    print("Training complete. Model saved to models/medical_cnn_weights.pth")

if __name__ == '__main__':
    # You will run this once you have downloaded the data!
    # train_model(data_dir='../data/chest_xray')
    print("Ready to train! Uncomment the function call when your data is in the data/ folder.")