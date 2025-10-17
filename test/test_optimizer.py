import random
from ..src.utils.optimizer import sgd
from ..src.utils.math import linreg, squared_loss
from ..src.utils.quick_gen import synthetic_data
import torch


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def test_sgd():
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # X和y的小批量损失
            l = loss(net(X, w, b), y)
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
