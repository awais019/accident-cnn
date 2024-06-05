import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset import train_loader, val_loader
from utils import train, validate

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)


optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


for epoch in range(10):
    train_loss, val_loss = train(
        model, train_loader, val_loader, optimizer, criterion, device)
    validation_accuracy = validate(model, val_loader, device)
    print(
        f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} Validation Accuracy: {validation_accuracy:.4f}')

torch.save(model.state_dict(), 'resnet18.pt')
