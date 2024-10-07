import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DROPOUT = 0.5

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = MNIST(root='./data', train=True, transform=transform, download=True)
test_data = MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        return x

model_FFNN = FFNN()
criterion = nn.MSELoss()
# optimizer = optim.SGD(model_FFNN.parameters(), lr=LEARNING_RATE)
# optimizer = optim.Adagrad(model_FFNN.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(model_FFNN.parameters(), lr=LEARNING_RATE)

def train(model, train_dataloader, epochs):
    model.train()

    for epoch in range(epochs):
        loss_in_epoch = 0

        for image, label in train_dataloader:
            output = model(image)
            one_hot_label = nn.functional.one_hot(label, 10).float()
            loss = criterion(output, one_hot_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_in_epoch += loss.item()

        total_loss = loss_in_epoch / len(train_dataloader)
        print(f'Epoch {epoch+1}/{epochs}; Loss: {total_loss:.4f}')

def evaluate(model, test_dataloader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for image, label in test_dataloader:
            output = model(image)
            pred = torch.argmax(output, dim=1)

            for i in range(len(label)):
                if label[i] == pred[i]:
                    correct += 1

    print(f'Correct: {correct}/{len(test_data)}')

train(model_FFNN, train_loader, EPOCHS)
evaluate(model_FFNN, test_loader)
