import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.cuda
from torch.utils.data import DataLoader
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time

import random
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt

## HOW TO FOR BASIC PYTORCH OPERATIONS.
## QUICK BOOT UP COVERING TENSORS, DATASETS AND NEURALNETS


# DATA must reside in a data folder within the project see below.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,  # not required now. You allready did this. In data folder
    transform=ToTensor()
)

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

# Iterators used to load data and reference them as a single object.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# is cuda available on this PC?
def torch_test():
    print(f'Cuda is available? -> {torch.cuda.is_available()}')
    x = torch.rand(5, 3)
    print(x)

# untested data import system
def dataimportcustom(Dataset):
    #Class
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def viewdatafiles(a_dataloader):
    # Display image and label.
    train_features, train_labels = next(iter(a_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

def dataimport_test():

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# Example of how to use pytorch tensors
def tensor_test():
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    tensor = torch.rand(3, 4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    # We move our tensor to the GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
    print(f"Now Device tensor is stored on: {tensor.device}")
    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:,1] = 0
    print(tensor)
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # ``tensor.T`` returns the transpose of a tensor
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    print(y2)
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    n = np.ones(5)
    t = torch.from_numpy(n)
    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Press the green button in the gutter to run the script.
def runNN():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    print(model)
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = torch.nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    input_image = torch.rand(3, 28, 28)
    print(input_image.size())
    flatten = torch.nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    layer1 = torch.nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = torch.nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")
    seq_modules = torch.nn.Sequential(
        flatten,
        layer1,
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10)
    )
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)
    softmax = torch.nn.Softmax(dim=1)
    pred_probab2 = softmax(logits)

    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


if __name__ == '__main__':
    print(torch.__file__)
    print_hi('PyCharm')
    torch_test()
    tensor_test()
    dataimport_test()
    #viewdatafiles(train_dataloader)
    # dataimportcustom(train_dataloader)
    # dataimportcustom.__init__()
    # trainingsize = dataimportcustom.len()
    # for i, data in enumerate(train_dataloader, 0):
    #     # get the inputs; data is a list of [inputs, labels]
    #     #inputs, labels = data
    #     viewdatafiles(data)
    runNN()
#
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
