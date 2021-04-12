import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Data Transforms
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Data Loader
train_path = 'dataset/training_data'
test_path = 'dataset/testing_data'

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
        self.conv_layer = nn.Sequential(  # input shape (3, 256, 256)
            nn.Conv2d(
                in_channels=3,  # rgb
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement
                padding=1,  # padding=(kernel_size-1)/2
            ),  # output shape (16, 256, 256)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # output shape (16, 128, 128)

            nn.Conv2d(16, 32, 3, 1, 1),  # output shape (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 64, 64)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(32 * 64 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 64 * 64)
        output = self.fc_layer(x)
        return output

# Print CNN
cnn = CNN()
print(cnn)

# Optimizer and loss function
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)   # optimize all cnn parameters, learning rate=0.001
loss_func = nn.CrossEntropyLoss()

best_accuracy = 0
for epoch in range(10):
    # Training
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
        train_accuracy = correct/total

    # Testing
    cnn.eval()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(test_loader):
        prediction = cnn(images)

        total += labels.size(0)
        _, predicted = torch.max(prediction.data, 1)
        correct += (predicted == labels).sum().item()
        test_accuracy = correct/total
    print('Epoch: ' + str(epoch+1) + ' Train Loss: ' + str(loss.item()) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(cnn.state_dict(), 'save.model')
        best_accuracy = test_accuracy