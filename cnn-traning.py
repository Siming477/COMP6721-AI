import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Transforms
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# DataLoader
train_path = 'Database/training-data'
test_path = 'Database/training-data'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=50, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=50, shuffle=True
)

# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 256, 256)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement
                padding=1,                  # padding=(kernel_size-1)/2
            ),                              # output shape (16, 256, 256)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 128, 128)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 128, 128)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 128, 128)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 64, 64)
        )
        self.out = nn.Linear(32 * 64 * 64, 3)   # fully connected layer, output 3 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 64 * 64)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

# Optimizer and loss function
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)   # optimize all cnn parameters, learning rate=0.001
loss_func = nn.CrossEntropyLoss()

# Training and Testing
for epoch in range(10):
    cnn.train()
    for i, (images, labels) in enumerate(train_loader):
        prediction = cnn(images)
        loss = loss_func(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = labels.size(0)
        _, predicted = torch.max(prediction.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct/total

    print('Epoch: '+str(epoch)+' Train Loss: '+str(loss.item())+' Train Accuracy: '+str(accuracy))

    cnn.eval()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(test_loader):
        prediction = cnn(images)

        total += labels.size(0)
        _, predicted = torch.max(prediction.data, 1)
        correct += (predicted == labels).sum().item()

    print("Test Accuracy: "+str(correct/total))