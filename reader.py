"""
File Reader API for UCI clustering benchmark.
reads the bin data and converts it to data and label np array
"""
import numpy as np
import zlib
import json

def read_uci(name):
	"""
	returns data and label np array having the name
	"""
	path = "./compressed/" + name + "/"
	path_data = path + "data.bin"
	path_labels = path + "labels.bin"
	## open the data and label binary file
	with open(path_data, 'rb') as f:
		data_comp = f.read()
	with open(path_labels, 'rb') as f:
		labels_comp = f.read()
	## convert the data and label to np array
	data = np.array(json.loads(zlib.decompress(data_comp).decode('utf8')))
	labels = np.array(json.loads(zlib.decompress(labels_comp).decode('utf8')))

	return data, labels

def read_uci_by_path(path):
	path_data = path + "data.bin"
	path_labels = path + "labels.bin"
	## open the data and label binary file
	with open(path_data, 'rb') as f:
		data_comp = f.read()
	with open(path_labels, 'rb') as f:
		labels_comp = f.read()
	## convert the data and label to np array
	data = np.array(json.loads(zlib.decompress(data_comp).decode('utf8')))
	labels = np.array(json.loads(zlib.decompress(labels_comp).decode('utf8')))

	return data, labels
