Due to the file limit of Github, the MNIST data can be found below:

  http://yann.lecun.com/exdb/mnist/

First download the data set by following the link above and rename the data folder to 'data'.

Run the 'MNIST_optimizer_training_path.py' file to get the data of the training process.
Then run the 'MNIST_optimizer_PCA.py' file to get a two-dimensional projection of the training process data.

```shell
python MNIST_optimizer_training_path.py
python MNIST_optimizer_PCA.py
```

It takes about 2 hours to compute the projection data on the NVIDIA RTX 3090 GPU. The expected output includes one-dimensional projection and two-dimensional projection results.
