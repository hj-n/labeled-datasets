import reader as rd
import numpy as np

data, label = rd.read_uci("svhn")

print(data.shape)