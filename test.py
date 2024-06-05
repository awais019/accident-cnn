import torch
from dataset import test_loader
from model import AccidentCNN


def test(model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the test images: {100 * correct // total}%')


model = AccidentCNN()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.load_state_dict(torch.load('accident_cnn.pt'))

model.to(device)

test(model, device)
