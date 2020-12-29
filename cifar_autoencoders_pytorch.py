import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), ])

train_dataset = CIFAR10(root='./data',
                        train=True,
                        download=True,
                        transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=16,
                          shuffle=True,
                          num_workers=2)

test_set = CIFAR10(root='./data',
                   train=False,
                   download=True,
                   transform=transform)

test_loader = DataLoader(test_set,
                         batch_size=16,
                         shuffle=False,
                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    np_img = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_model(epochs, autoencoder, criterion, optimizer):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    print('Saving Model...')
    torch.save(autoencoder.state_dict(), MODEL_PATH)


def evaluate(autoencoder):
    print("Loading checkpoint...")
    autoencoder.load_state_dict(torch.load(MODEL_PATH))
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(5)))
    imshow(torchvision.utils.make_grid(images))

    images = Variable(images.cuda())

    decoded_imgs = autoencoder(images)[1]
    imshow(torchvision.utils.make_grid(decoded_imgs.data))


if __name__ == '__main__':
    EPOCHS = 10
    MODEL_PATH = './models/cifar_autoencoder.pkl'
    Train = True  # Set to false if already trained

    autoencoder = AutoEncoder().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    if Train:
        print("============== Training ==============")
        train_model(epochs=EPOCHS, autoencoder=autoencoder, criterion=criterion, optimizer=optimizer)
    print("============== Evaluate ==============")
    evaluate(autoencoder)
