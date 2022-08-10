# 96 Labeled Datasets

In this repository, we provide 96 publicly available labeled dataset.
The datasets were originally collected to be utilized in the paper "Sanity Check for External Clustering Validation Benchmarks using Internal Validation Measure", as a potential candidate for external clustering validation. However, it sill can be used for various purposes (e.g., classification, dimensionality reduction, etc.) For better applicability, we provide datasets in both numpy (`.npy`) and compressed (`.bin`) format. We also provided a reader code for the compressed files.

### Reader API

#### API

The reader of the compressed files are written in `reader.py`. We assume that the relative path of the reader file and the compressed datasets is identical to the one of this repository. The reader code depends on `numpy` and `zlib`.

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


### Summary Statistics

Our original purpose of gathering the datasets is to measure and compare Class-Label Matching (CLM; the degree of how the class labels of a certain dataset matches with its cluster structure) of the datasets, so that we can reliably use the datasets with high CLM for external clustering validation.
The description about measuring and comparing CLM is well described in our paper "Sanity Check for External Clustering Validation Benchmarks using Internal Validation Measures". 

Our summary statatics, which is stored in `summary.csv`, not only contains the basic info of the datasts but also the CLM scores computed by various measures, which are listed in our paper. The explanation about each column of the file is as follows.

- `dataset`: the name of the dataset
- `objects`: objects # (i.e., data points) 
- `features`: features # (i.e., attributes, dimensions)
- `labels`: labels # (i.e., classes)
- `ch_btw`: CLM score computed by [Between-dataset Calinski-Harabasz index](https://github.com/hj-n/btw-dataset-internal-measures)
  - Note that between-dataset Calinski-Harabasz index is the one proposed in our paper
- `ch`: CLM score computed by Calinski-Harabasz index
- `db`: CLM score computed by Davies-Bouldin index
- `dunn`: CLM score computed by Dunn index
- `ii`: CLM score computed by I Index
- `sil`: CLM score computed by Silhouette coefficient
- `xb`: CLM score computed by Xie-Beni Index
- `knn`: CLM score computed by K-Nearest Neighbor Classifier
- `nb`: CLM score computed by Naive Bayes Classifier
- `rf`:  CLM score computed by Random Forest Classifier
- `lr`: CLM score computed by Logistic Regression Classifier
- `lda`: CLM score computed by Linear Discriminant Analysis Classifier
- `mlp`: CLM score computed by Multilayer Perceptron Classifier
- `ensemble_classifiers`: CLM score computed by the ensemble of the classifiers
- `{clustering_algo}_{ext_measure}`: CLM score computed by the combintation of {clustering_algo} (clustering algorithm) and {ext_measure} (external clustering validation measure)
  - `{clustering_algo}` can be 
    - `agglo_average`: [Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) with average linkage
    - `agglo_single`: [Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) with single linkage
    - `agglo_complete`: [Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) with complete linkage
    - `birch`: Birch clustering algorithm
    - `dbscan`: DBSCAN clustering algorithm
    - `hdbscan`: [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) clustering algorithm
    - `kmeans`: *K*-Means clustering algorithm
    - `xmeans`: *X*-Means clustering algorithm
    - `kmedoid`: *K$-Medoid clustering algorithm
  - `{ext_measure}` can be
    - `ami`: Adjusted Mutual Information score
    - `arand`: Adjusted Rand Index
    - `vm`: V-measure score
    - `nmi`: Normalized Mutual Information score



### Contact

If you have any issue exploiting the datasets, feel free to contact us via [hj@hcil.snu.ac.kr](mailto:hj@hcil.snu.ac.kr).

### Reference

TBA
