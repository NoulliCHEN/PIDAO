Due to the file limit of Github, the data can be found below:

  1. MNIST: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/, 
 
  2. FashionMNIST: https://www.kaggle.com/datasets/zalando-research/fashionmnist, 
  
  3. CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html.

After creating a new folder named 'data', one follows the links above to download the dataset to the 'data' folder.

For MNIST: 
```shell
python MNIST_optimizer.py
```
It takes about one and a half hours to train the neural network on the NVIDIA RTX 3090 GPU. The expected output of this demo contains 'Train Loss', 'Valid Loss', 'Train Acc.' and 'Valid Acc.'.

For FashionMNIST:
```shell
python FashionMNIST_optimizer.py
```
It takes about one and a half hours to train the neural network on the NVIDIA RTX 3090 GPU. The expected output of this demo contains 'Train Loss', 'Valid Loss', 'Train Acc.' and 'Valid Acc.'.

For Cifar10:
```shell
python run Cifar10_optimizer.py
```
It takes about three hours to train the neural network on the NVIDIA RTX 3090 GPU. The expected output of this demo contains 'Train Loss', 'Valid Loss', 'Train Acc.' and 'Valid Acc.'.
