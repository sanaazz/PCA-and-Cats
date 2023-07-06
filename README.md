# PCA-and-Cats

In this code, we analyze cat images using PCA. we implement the PCA algorithm by basic libraries.
Before the analysis, we resize the images to 64 × 64 pixels by using the bilinear interpolation method3 implemented in the PIL library4. Then,we  flatten all images of size 64 × 64 × 3 to obtain a 4096 × 3 matrix for each image.

Note that all images are 3-channel RGB. we create a 3-D array, X, of size 5653 × 4096 × 3 by stacking flattened matrices of the images provided in the dataset. Then, we slice X as Xi = X[:, :, i], where i corresponds to the three indexes (0: Red, 1: Green, and 2: Blue), to obtain color channel matrix (5653 × 4096) of all images for each channel.
