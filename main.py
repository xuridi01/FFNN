import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, CIFAR10

transform = transforms.Compose([
    transforms.ToTensor(),
])

BATCH_SIZE = 64

def load_mnist():
    train_data = MNIST(root='./data', train=True, transform=transform, download=True)
    test_data = MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, len(test_data)

def load_cifar10():
    train_data = CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, len(test_data)

class FFNN(nn.Module):
    def __init__(self, activation_fun, input_neurons, output_neurons, hidden1, hidden2):
        super(FFNN, self).__init__()
        self.activation_fun = activation_fun
        self.fc1 = nn.Linear(input_neurons, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_neurons)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation_fun(self.fc1(x))
        x = self.activation_fun(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, optim, train_dataloader, epochs):
    model.train()

    for epoch in range(epochs):
        with open('act_fun_log.txt', 'a') as act_fun_log:
            loss_in_epoch = 0

            for image, label in train_dataloader:
                output = model(image)
                loss = criterion(output, label)
                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_in_epoch += loss.item()

            total_loss = loss_in_epoch / len(train_dataloader)
            act_fun_log.write(f'Epoch {epoch+1}/{epochs}; Loss: {total_loss:.4f}\n')

def evaluate(model, test_dataloader, len_test_d):
    model.eval()
    correct = 0

    with torch.no_grad():
        for image, label in test_dataloader:
            output = model(image)
            pred = torch.argmax(output, dim=1)

            for i in range(len(label)):
                if label[i] == pred[i]:
                    correct += 1

    with open('act_fun_log.txt', 'a') as act_fun_log:
        act_fun_log.write(f'Correct: {correct}/{len_test_d}\n')

criterion = nn.CrossEntropyLoss()
activation_fun_list = [nn.functional.relu, nn.functional.sigmoid, nn.functional.elu, nn.functional.selu, nn.functional.gelu, nn.functional.silu]
activation_fun_list_names = ['RELU', 'SIGMOID', 'ELU', 'SELU', 'GELU', 'SILU']

with open('act_fun_log.txt', 'w') as act_fun_log:
    act_fun_log.write(f'Experiments with different activation functions on datasets MNIST and CIFAR-10\n')

#MNIST
EPOCHS = 10
LEARNING_RATE = 0.001
train_loader, test_loader, len_test_data = load_mnist()

with open('act_fun_log.txt', 'a') as act_fun_log:
    act_fun_log.write(f'---MNIST---\n\n')
    act_fun_log.write(f'Starting params:\n-Epochs: {EPOCHS}\n-Learning rate: {LEARNING_RATE}\n-Batch size: {BATCH_SIZE}\n-Optimizer: ADAM\n-Loss function: CrossEntropy\n\n')

for fun, name in zip(activation_fun_list, activation_fun_list_names):
    with open('act_fun_log.txt', 'a') as act_fun_log:
        act_fun_log.write(f'\n{name}:\n\n')
    model_FFNN = FFNN(fun, 784, 10, 256, 128)
    train(model_FFNN, optim.Adam(model_FFNN.parameters(), lr=LEARNING_RATE), train_loader, EPOCHS)
    evaluate(model_FFNN, test_loader, len_test_data)

#CIFAR
EPOCHS = 10
LEARNING_RATE = 0.005
train_loader, test_loader, len_test_data = load_cifar10()

with open('act_fun_log.txt', 'a') as act_fun_log:
    act_fun_log.write(f'\n\n---CIFAR-10---\n\n')
    act_fun_log.write(f'Starting params:\n-Epochs: {EPOCHS}\n-Learning rate: {LEARNING_RATE}\n-Batch size: {BATCH_SIZE}\n-Optimizer: ADAM\n-Loss function: CrossEntropy\n\n')

for fun, name in zip(activation_fun_list, activation_fun_list_names):
    with open('act_fun_log.txt', 'a') as act_fun_log:
        act_fun_log.write(f'\n{name}:\n\n')
    model_FFNN = FFNN(fun, 3072, 10, 1024, 512)
    train(model_FFNN, optim.Adam(model_FFNN.parameters(), lr=LEARNING_RATE), train_loader, EPOCHS)
    evaluate(model_FFNN, test_loader, len_test_data)

