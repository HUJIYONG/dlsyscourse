import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    f = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )

    return nn.Sequential(
        nn.Residual(f),
        nn.ReLU(),
    )



def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )






def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is not None:
        model.train()
    else:
        model.eval()

    loss_func = nn.SoftmaxLoss()

    losses = []
    total_acc = 0

    for X, y in dataloader:
        pred = model(X)
        loss = loss_func(pred, y)

        losses.append(loss.numpy())

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

        total_acc += (pred.numpy().argmax(axis=1) == y.numpy()).sum()

    return 1 - total_acc / len(dataloader.dataset), np.array(losses).mean()



def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)

    data_set = ndl.data.MNISTDataset(
        image_filename=os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        label_filename=os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
    )

    data_loader = ndl.data.DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLPResNet(
        dim=28 * 28,
        hidden_dim=hidden_dim,
        num_blocks=3,
        num_classes=10
    )

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_err, train_loss = 0, 0
    test_err, test_loss = 0, 0

    for _ in range(epochs):
        start_time = time.time()
        train_err, train_loss = epoch(data_loader, model, opt)
        test_err, test_loss = epoch(data_loader, model, None)
        end_time = time.time()
        print(f"Epoch {_ + 1}/{epochs}, Train Error: {train_err:.4f}, Train Loss: {train_loss:.4f}, Test Error: {test_err:.4f}, Test Loss: {test_loss:.4f}, Time: {end_time - start_time:.2f}s")


    return train_err, train_loss, test_err, test_loss
        


if __name__ == "__main__":
    train_mnist(data_dir="../data")
