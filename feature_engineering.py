import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# method 2: auto with PCA
def pca_reduce(x_train, x_test):
    pca = PCA(n_components = 0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative variance:", pca.explained_variance_ratio_.cumsum())
    return x_train_pca, x_test_pca

# manuall reduces columns by looking at variance and correlation
def manual_reduce(x_train, x_test):
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    # check correlation and find redundant columns
    #num_cols = x_train.select_dtypes(include='number')
    corr = x_train.corr()

    #upper triangle of corr
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_threshold = 0.95

    corr_drop = [
        column for column in upper.columns
        if any(upper[column] > corr_threshold)
    ]

    x_train = x_train.drop(columns=corr_drop)
    x_test = x_test.drop(columns=corr_drop)

    print(f"Dropping cols with high corr: {corr_drop}")

    # discard low_variance columns
    variance = x_train.var()
    var_threshold = 1e-3
    low_var_cols = variance[variance < var_threshold].index.tolist()

    print(f"dropping low var cols: {low_var_cols}")

    x_train = x_train.drop(columns=low_var_cols, errors="ignore")
    x_test = x_test.drop(columns=low_var_cols, errors="ignore")

    return x_train, x_test

#mode: manual, PCA, both
# manual - manual feature reduction
# PCA - use PCA
def reduce_dim(x_train, x_test, mode="pca", standard=None):
    # standardize
    x_train = np.log1p(x_train)
    x_test = np.log1p(x_test)
    scaler = preprocessing.RobustScaler(quantile_range=(10, 90))
    if standard:
        scaler = preprocessing.StandardScaler()

    print(f"Original shape: {x_train.shape}")
    if mode == "PCA":
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train, x_test = pca_reduce(x_train, x_test)

    if mode == "manual":
        x_train, x_test = manual_reduce(x_train, x_test)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    print(f"Reduced shape: {x_train.shape}")
        

    return x_train, x_test


def visualize_samples(samples, labels, title="Samples visualization", colors=["green", "orange", "red", "yellow","blue","pink", "purple", "grey"]):
    """ Visualize first three dimensions of samples. """
    # Convert colors to NumPy array
    colors = np.array(colors)
    # Create figure
    fig = plt.figure(figsize=(20, 5))

    unique_labels = np.unique(labels)
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    labels_int = np.array([label_to_int[l] for l in labels])


    # Plot first two dimensions on a 2D plot
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_2d.set_title(f"{title} [2D]")
    ax_2d.scatter(samples[:, 0], samples[:, 1], c=colors[labels_int])

    print(fig)