import reader as rd
import numpy as np

data, label = rd.read_dataset("cifar10")

print(data.shape)