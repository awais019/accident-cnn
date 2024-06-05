import torch
import torch.nn as nn
import torch.optim as optim


from model import AccidentCNN
from dataset import train_loader, val_loader
from utils import train, validate


model = AccidentCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Training on device {device}.")

model.to(device)


for epoch in range(10):
    train_loss, val_loss = train(
        model, train_loader, val_loader, optimizer, criterion, device)
    validation_accuracy = validate(model, val_loader, device)
    print(
        f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} Validation Accuracy: {validation_accuracy:.4f}')


torch.save(model.state_dict(), "accident_cnn.pt")
print("Model saved.")
