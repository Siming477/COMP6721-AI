import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Data Transforms
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Test data Loader
test_path = 'dataset/testing_data'
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

cnn = CNN()

# Load model
load_cnn = torch.load('save.model')
cnn = CNN()
cnn.load_state_dict(load_cnn)

# Testing
cnn.eval()
total = 0
correct = 0
for i, (images, labels) in enumerate(test_loader):
    prediction = cnn(images)

    total += labels.size(0)
    _, predicted = torch.max(prediction.data, 1)
    correct += (predicted == labels).sum().item()
print("Test Accuracy: "+str(correct/total))

# Evaluation
y_true = []
y_pred = []
target_names = ['not_person', 'with_mask', 'without_mask']
for data, target in test_loader:
    for label in target.data.numpy():
        y_true.append(label)
    for prediction in cnn(data).data.numpy().argmax(1):
        y_pred.append(prediction)
print("Classification Report: ")
print(classification_report(y_true, y_pred, target_names=target_names))
print("Confusion Matrix: ")
print(confusion_matrix(y_true, y_pred))




