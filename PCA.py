import os.path

import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import BILINEAR
from PIL import Image
from os import getcwd, rename
import pandas as pd


def rename_flickr_img(path):
    """
    rename the images in the dataset folder in a way that the numbering starts from 1 and ends at 821.
    """
    if not os.path.exists(path + "/1.jpg"):
        for i in range(2, 822):
            if i < 10:
                rename(path + '/flickr_cat_00000' + str(i) + '.jpg', path + '/' + str(i - 1) + '.jpg')
            elif i < 100:
                rename(path + '/flickr_cat_0000' + str(i) + '.jpg', path + '/' + str(i - 1) + '.jpg')
            else:
                rename(path + '/flickr_cat_000' + str(i) + '.jpg', path + '/' + str(i - 1) + '.jpg')


def rename_pixabay_img(path):
    """
    rename the images in the dataset folder in a way that the numbering starts from 1 and ends at 821.
    """
    if not os.path.exists(path + "/821.jpg"):
        for i in range(2, 4835):
            if i < 10:
                rename(path + '/pixabay_cat_00000' + str(i) + '.jpg', path + '/' + str(i + 819) + '.jpg')
            elif i < 100:
                rename(path + '/pixabay_cat_0000' + str(i) + '.jpg', path + '/' + str(i + 819) + '.jpg')
            elif i < 1000:
                rename(path + '/pixabay_cat_000' + str(i) + '.jpg', path + '/' + str(i + 819) + '.jpg')
            else:
                rename(path + '/pixabay_cat_00' + str(i) + '.jpg', path + '/' + str(i + 819) + '.jpg')


def resizing(img, size=(64, 64)):
    """
    resize the images to 64 × 64 pixels by using the bilinear interpolation method3 implemented in the PIL
    library.
    flatten all images of size 64 × 64 × 3 to obtain a 4096 × 3 matrix for each image.
    the PIL library reads the image files in uint8 format. converting the  data type to int or float32.
    all images are 3-channel RGB. Create a 3-D array, X, of size 5653 × 4096 ×
    3 by stacking flattened matrices of the images provided in the dataset.
    """
    img = img.resize(size, BILINEAR)
    img = np.array(img)
    # convert the data type to int or float32
    img = img.astype(np.int32)
    return img.reshape(-1, 3)


def min_max_scaling(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class PCA:
    """
       PCA class to calculate the first n principal components of images for each channel: shape(5653, 4096, 1)
       """

    def __init__(self):
        self.n_components = None
        self.mean = None
        self.eigen_vecs = None
        self.eigen_vals = None

    def fit(self, x):
        # mean centering
        self.mean = np.mean(x, axis=0)
        x = x - self.mean

        # covariance
        cov = np.cov(x.T)

        # eigenvectors , eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose this for easier calculation
        eigenvectors = eigenvectors.T

        # sort the eigenvectors according to eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        self.eigen_vals = eigenvalues[idxs]
        self.eigen_vecs = eigenvectors[idxs]

        # choose first k eigenvectors
        # self.eig = eigenvectors[:self.n_components]

    def transform(self, x, n_components):
        self.n_components = n_components
        # project data, Transform the X features based on the found Eigen Values and vectors
        x = x - self.mean
        return np.dot(x , self.eigen_vecs[:n_components, :].T)

    def pve(self):
        """
        calculate the proportion of variance explained by the nth component
        :return: the proportion of variance explained by the nth component
        """
        return self.eigen_vals[self.n_components - 1] / np.sum(self.eigen_vals)

    def cpve(self, percent=None):
        """
        calculate the cumulative proportion of variance explained by the first n components
        :return: the cumulative proportion of variance explained by the first n components
        """
        if percent is None:
            return np.sum(self.eigen_vals[:self.n_components]) / np.sum(self.eigen_vals)

        for i in range(len(self.eigen_vals)):
            if np.sum(self.eigen_vals[:i]) / np.sum(self.eigen_vals) >= percent:
                print(
                    f"The cumulative proportion of variance explained by the first {i} components "
                    f" has {percent * 100} %: PVE.", np.sum(self.eigen_vals[:i]) / np.sum(self.eigen_vals))
                break

    def get_eigenvectors(self):
        return self.eigen_vecs

    def get_mean(self):
        return self.mean


def main():
    """
    main function
    """
    path = f'{getcwd()}/afhq_cat'
    rename_flickr_img(path)
    rename_pixabay_img(path)
    # rename_flickr_img() and rename_pixabay_img() are used to rename the images in the dataset folder in a way that
    # the numbering starts from 1 and ends at 5653.
    # resize the images to 64 × 64 pixels and flatten all images to obtain a 4096 × 3 matrix for each image

    images = []
    for i in range(1, 5654):
        img = Image.open(path + '/' + str(i) + '.jpg')
        img = resizing(img)
        images.append(img)
    images = np.array(images)

    plt.imshow(images[1].reshape(64, 64, 3))
    plt.savefig('cat.png')
    # Question 1.1
    # Report PVE and CPVE for the first 10 principal components of each channel
    # report the least number of principal components that explain 70% of the variance for each channel

    pca_ch1 = PCA()
    pca_ch2 = PCA()
    pca_ch3 = PCA()
    pca_ch1.fit(images[:, :, 0])
    pca_ch2.fit(images[:, :, 1])
    pca_ch3.fit(images[:, :, 2])

    results = pd.DataFrame(columns=['n_components', 'Channel1', 'Channel2', 'Channel3'])

    cmp = []
    ch0 = []
    ch1 = []
    ch2 = []
    for i, item in enumerate([pca_ch1, pca_ch2, pca_ch3]):
        for j in range(1, 11):
            x_projected = item.transform(images[:, :, i], j)
            # print("Shape of X for channel", i, ":", images[:, :, i].shape)
            # print("Shape of transformed X for channel", i, ":", x_projected.shape)
            print(f"PVE of {j}th component for channel", i+1, ":", item.pve())
            print(f"cumulative PVE until {j}th component is: ", item.cpve())
            print("--------------------------------------------------")
            cmp.append(j)
            if i == 0:
                ch0.append(item.pve())
            elif i == 1:
                ch1.append(item.pve())
            else:
                ch2.append(item.pve())
        item.cpve(0.70)
    cmp = set(cmp)
    results['n_components'] = list(cmp)
    results['Channel1'] = ch0
    results['Channel2'] = ch1
    results['Channel3'] = ch2
    results.to_excel('results.xlsx', index=False)

    # Question 1.2
    # plot first 10 eigenfaces
    eigenvectors_ch1 = pca_ch1.get_eigenvectors()
    eigenvectors_ch2 = pca_ch2.get_eigenvectors()
    eigenvectors_ch3 = pca_ch3.get_eigenvectors()

    row = 2
    col = 5
    for i in range(10):
        pic1_ch1 = min_max_scaling(eigenvectors_ch1[i].reshape(64, 64))
        pic1_ch2 = min_max_scaling(eigenvectors_ch2[i].reshape(64, 64))
        pic1_ch3 = min_max_scaling(eigenvectors_ch3[i].reshape(64, 64))

        pic = np.stack([pic1_ch1, pic1_ch2, pic1_ch3], axis=2)

        plt.subplot(row, col, i + 1)
        plt.imshow(pic)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.title(f'k={i+1}')
    plt.savefig('eigenfaces.png')
    plt.show()

    # Question 1.3
    # reconstruct the second image  with k components
    col = 3
    row = 2
    k = [1, 50, 250, 500, 1000, 4096]
    for j, i in enumerate(k):
        x_projected_ch1 = pca_ch1.transform(images[:, :, 0], i)
        x_projected_ch2 = pca_ch2.transform(images[:, :, 1], i)
        x_projected_ch3 = pca_ch3.transform(images[:, :, 2], i)

        x_reconstructed_ch1 = np.dot(x_projected_ch1, pca_ch1.get_eigenvectors()[:i, :])
        x_reconstructed_ch2 = np.dot(x_projected_ch2, pca_ch2.get_eigenvectors()[:i, :])
        x_reconstructed_ch3 = np.dot(x_projected_ch3, pca_ch3.get_eigenvectors()[:i, :])

        x_reconstructed_ch1 += pca_ch1.get_mean()
        x_reconstructed_ch2 += pca_ch2.get_mean()
        x_reconstructed_ch3 += pca_ch3.get_mean()

        pic1_ch1 = min_max_scaling(x_reconstructed_ch1[1].reshape(64, 64))
        pic1_ch2 = min_max_scaling(x_reconstructed_ch2[1].reshape(64, 64))
        pic1_ch3 = min_max_scaling(x_reconstructed_ch3[1].reshape(64, 64))

        pic = np.stack([pic1_ch1, pic1_ch2, pic1_ch3], axis=2)

        plt.subplot(row, col, j + 1)
        plt.imshow(pic)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.title(f'k={i}')
    plt.savefig('reconstructed.png')
    plt.show()

    plt.imshow(pic)
    plt.savefig('pic_reconstructed.png')


if __name__ == "__main__":
    main()
