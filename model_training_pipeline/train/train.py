import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
from data.dataset import get_data_loaders
from models.resnet_model import ResNet18Model

def train_model(data_dir, num_epochs, batch_size, learning_rate, save_path, validation_data_loader):
    data_loader = get_data_loaders(data_dir, batch_size)
    model = ResNet18Model(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 4 == 0:
            torch.save(model.state_dict(), f'{save_path}_epoch_{epoch + 1}.pt')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_data_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(validation_data_loader)
        accuracy = 100.0 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}] | Training Loss: {loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {accuracy:.2f}%')

        training_losses.append(loss)
        validation_losses.append(val_loss)

    torch.save(model.state_dict(), save_path)

    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_graph.png')
    plt.show()
