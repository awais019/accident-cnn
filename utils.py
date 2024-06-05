import torch


def train(model, trainloader, valloader, optimizer, criterion, device):
    model.train()
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

    return train_loss.item(),  val_loss.item()


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
