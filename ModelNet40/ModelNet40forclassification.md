Due to the file limit of Github, the data can be found below:

  1. ModelNet40: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip,

After creating a new folder named 'data', one follows the links above to download the dataset to the 'data' folder.

For ModelNet40: 
```shell
cd utils
python train_classification.py
```
It takes about one hour to train the neural network on the NVIDIA Quadro RTX 6000 GPU. The expected output of this demo contains 'Train Loss', 'Valid Loss', 'Train Acc.' and 'Valid Acc.'.

