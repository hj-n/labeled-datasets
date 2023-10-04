# 96 Labeled Datasets

In this repository, we provide 96 publicly available labeled dataset.
The datasets were originally collected to be utilized in the paper "Measuring the Validity of Clustering Validation Datasets", previously entitled "Sanity Check for External Clustering Validation Benchmarks using Internal Validation Measure", as a potential candidate for external clustering validation. However, it sill can be used for various purposes (e.g., classification, dimensionality reduction, etc.) For better applicability, we provide datasets in both numpy (`.npy`) and compressed (`.bin`) format. We also provided a reader code for the compressed files.

A full list of the datasets is available at [this website](https://hyeonword.com/clm-datasets/) and the Appendix of our reference paper (TBA). 

### Reader API

#### API

The reader of the compressed files is written in `reader.py`. We assume that the relative path of the reader file and the compressed datasets is identical to the one of this repository. The reader code depends on `numpy` and `zlib`.

> `read_dataset(name)`
> - returns the designated dataset as a form of numpy arrays holding the data and labels
> > - (INPUT) `name`: str, the name of a dataset (directory name)
> > - (OUTPUT) `(data, labels)`: 
> >   - `data`: ndarray, 2D numpy array holding the data values
> >   - 'label`: ndarray, 1D numpy array holding the class labels 

> `read_dataset_by_path(path)`
> - returns the designated dataset as a form of numpy arrays holding the data and labels
> > - (INPUT) `path`: str, the relative path to a directory containing the datasets
> > - (OUTPUT) `(data, labels)`: identical to `read_dataset`

> `read_multiple_datasets(names)`
> - returns the dictionary holding the data and labels of the designated datasets
> > - (INPUT) 'names': list, the list holding the names of datasets
> > - (OUTPUT) `(data, labels)`:
> >   - `data`: dict, dictionary holding the data values; the value of a certain dataset can be accessed by using the name of the dataset as a key
> >   - `labels`: dict, dictionary holding the labels; the label of a certain dataset can be accessed by using the name of the dataset as a key

> `read_all_datasets()`
> - returns the dictionary tholding the data and labels of entire 96 datasets
> > - (OUTPUT) `(data, labels)`: identical to `read_multiple_datasets`

#### Example

```python3
import reader as rd
import numpy as np

data, label = rd.read_dataset("cifar10")
```

### Contact

If you have any issue exploiting the datasets, feel free to contact us via [hj@hcil.snu.ac.kr](mailto:hj@hcil.snu.ac.kr).

### Reference

TBA
