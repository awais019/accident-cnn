import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

model = models.vgg13(models.VGG13_Weights.DEFAULT)

for param in model.features.parameters():
    param.requires_grad = False

num_features = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(
    root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(
    root='dataset/val', transform=transform)
test_dataset = datasets.ImageFolder(
    root='dataset/test', transform=transform)

trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=100, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return loss.item()


def validate(model, validationloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in validationloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


for epoch in range(10):
    train_loss = train(model, trainloader, optimizer, criterion, device)
    validation_accuracy = validate(model, valloader, device)
    print(
        f'Epoch {epoch+1}, Loss: {train_loss}, Validation Accuracy: {validation_accuracy}')
