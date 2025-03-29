"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename, 'rb') as f:
        # read image file header
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32, count=4).byteswap()
        # read image data
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols).astype(np.float32)
        # normalize image data
        X = X / 255.0

    with gzip.open(label_filename, 'rb') as f:
        # read label file header
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.uint32, count=2).byteswap()
        # read label data
        y = np.frombuffer(f.read(), dtype=np.uint8)

    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    num_examples = Z.shape[0]
    num_classes = Z.shape[1]
    exp_Z = ndl.exp(Z)
    sum_exp_Z = ndl.summation(exp_Z, axes=(1,))
    log_sum_exp_Z = ndl.log(sum_exp_Z)
    loss = -ndl.summation(Z * y_one_hot) + ndl.summation(log_sum_exp_Z)
    return loss / num_examples


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    num_batches = num_examples // batch
    for i in range(num_batches):
        X_batch = X[i * batch : (i + 1) * batch]
        y_batch = y[i * batch : (i + 1) * batch]

        # compute loss
        X_batch = ndl.Tensor(X_batch)
        Z1 = ndl.matmul(X_batch, W1)
        A1 = ndl.relu(Z1)
        Z2 = ndl.matmul(A1, W2)

        y_one_hot = np.zeros((batch, num_classes))
        y_one_hot[np.arange(batch), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        loss = softmax_loss(Z2, y_one_hot)
        loss.backward()

        new_W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        new_W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())

        (W1, W2) = (new_W1, new_W2)

    return (W1, W2)






### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
