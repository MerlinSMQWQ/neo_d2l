import torchvision
from torchvision import transforms
from torch.utils import data
import torch
from torch import nn
from ..src.utils.quick_gen import synthetic_data
from ..src.utils.data_loader import load_array, save_images, get_fashion_mnist_labels

def test_load_array():
    true_w = torch.tensor([2.0, -5.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    
    data_iter = load_array((features, labels), 100, is_train=True)
    
    net = nn.Sequential(nn.Linear(2, 1))
    
    loss = nn.MSELoss()
    
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    
    num_epochs = 3
    
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss = {l}")
        
    w = net[0].weight.data
    assert isinstance(w, torch.Tensor)
    print('w的估计误差:', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    assert isinstance(b, torch.Tensor)
    print('b的估计误差:', true_b - b)
    
    
def test_fashion_mnist():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root=r"./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=r"./data", train=False, transform=trans, download=True)
    print(len(mnist_train), len(mnist_test))     
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    save_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y), png_path= r'./lab_img/output.png');
    