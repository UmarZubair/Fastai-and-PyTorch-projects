import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader as DataLoader

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_set = CIFAR10(root='./data',
                    train=True,
                    download=True,
                    transform=transform_train)

training_loader = DataLoader(dataset=train_set,
                             batch_size=128,
                             shuffle=True,
                             num_workers=2)

validation_set = CIFAR10(root='./data',
                         train=False,
                         download=True,
                         transform=transform_train)

validation_loader = DataLoader(dataset=validation_set,
                               batch_size=100,
                               shuffle=False)

test_set = CIFAR10(root='./data',
                   train=False,
                   download=True,
                   transform=transform_test)

test_loader = DataLoader(dataset=test_set,
                         batch_size=100,
                         shuffle=False,
                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def train(epochs,lr,weight_decay):
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_history = []
    correct_history = []
    val_loss_history = []
    val_correct_history = []
    for e in range(epochs):
        loss = 0.0
        correct = 0.0
        val_loss = 0.0
        val_correct = 0.0
        for input, labels in training_loader:
            input = input.to(device)
            labels = labels.to(device)
            outputs = model(input)
            loss1 = criterion(outputs, labels)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            _, predictions = torch.max(outputs, 1)
            loss += loss1.item()
            correct += torch.sum(predictions == labels.data)
        else:
            with torch.no_grad():
                for val_input, val_labels in validation_loader:
                    val_input = val_input.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_input)
                    val_loss1 = criterion(val_outputs, val_labels)
                    _, val_preds = torch.max(val_outputs, 1)
                    val_loss += val_loss1.item()
                    val_correct += torch.sum(val_preds == val_labels.data)
            epoch_loss = loss / len(training_loader)
            epoch_acc = correct.float() / len(training_loader)
            loss_history.append(epoch_loss)
            correct_history.append(epoch_acc)
            val_epoch_loss = val_loss / len(validation_loader)
            val_epoch_acc = val_correct.float() / len(validation_loader)
            val_loss_history.append(val_epoch_loss)
            val_correct_history.append(val_epoch_acc)
            print('epoch :', (e + 1))
            print('training loss: {:.4f}, accuracy {:.4f} '.format(epoch_loss, epoch_acc.item()))
            print('validation loss: {:.4f}, validation accuracy {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

    print('Finished Training')
    print('Saving Model...')
    torch.save(model.state_dict(), MODEL_PATH)


def test():
    correct = 0
    test_loss = 0
    total = 0
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(MODEL_PATH))
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images, labels = data[0], data[1]
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
            epoch_loss = loss / len(test_loader)
            epoch_acc = correct.float() / len(test_loader)
        print('testing loss: {:.4f}, test accuracy {:.4f} '.format(epoch_loss, epoch_acc.item()))


if __name__ == '__main__':
    MODEL_PATH = './models/cifar_lenet.pth'
    EPOCHS = 1
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4

    print("============== Training ==============")
    train(epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print("============== Testing ==============")
    test()
