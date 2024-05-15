"""
Calculates the Fr'echet Distance between two multivariate Gaussians fit to two groups of embeddings.
"""

# Imports
import argparse
import numpy as np
import os
import tqdm
from scipy import linalg

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('-f1', '--feat_dir1', type=str, required=True, help='Specify the path to the folder that contains the first group of embeddings.')
required.add_argument('-f2', '--feat_dir2', type=str, required=True, help='Specify the path to the folder that contains the second group of embeddings.')
args = parser.parse_args()

# Functions
def calculate_activation_statistics(f_dir):
    """
    Calculates the mean and standard deviation of a set of features.

    Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    Inputs:
        f_dir (str): The directory that contains the features to be evaluated.
    Returns:
        mu (ndarray): Sample mean of features.
        sigma (ndarray): Coviarance of features.
    """
    pred_arr = []
    for feature_f in tqdm.tqdm(os.listdir(f_dir)):
        pred_arr.append(np.load(os.path.join(f_dir, feature_f)))
    if np.sum(np.isnan(pred_arr)) > 0:
        print("NAN present", np.sum(np.isnan(pred_arr)))
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    Taken from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py

    Inputs:
        mu1 (ndarray): The sample mean over the first set of features.
        mu2 (float) : The sample mean over the second set of features.
        sigma1 (float): The covariance matrix over the first set of features.
        sigma2 (float): The covariance matrix over the second set of features.
    Returns:
        (float): The Frechet distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# Main Code
if __name__ == "__main__":
    mu1, sigma1 = calculate_activation_statistics(args.feat_dir1)
    mu2, sigma2 = calculate_activation_statistics(args.feat_dir2)
    print(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))
