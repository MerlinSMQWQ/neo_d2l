import torch
from torch import nn
from ..src.utils.quick_gen import synthetic_data
from ..src.utils.data_loader import load_array

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