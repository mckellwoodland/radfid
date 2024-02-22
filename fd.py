"""
Calculates the Fr'echet Distance between two multivariate Gaussians fit to two groups of embeddings.
"""

# Imports
import argparse
import numpy as np
import os

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('-f1', '--feat_dir1', type=str, required=True, help='Specify the path to the folder that contains the first group of embeddings.')
required.add_argument('-f2', '--feat_dir2', type=str, required=True, help='Specify the path to the folder that contains the second group of embeddings.')
args = parser.parse_args()

# Main Code
if __name__ == "__main__":
    for feature_f in os.listdir(args.feat_dir1)[:10]:
        feat1 = np.load(os.path.join(args.feat_dir1, feature_f))
        print(len(feat1))
        print(feat1[:10])
        feat2 = np.load(os.path.join(args.feat_dir2, feature_f))
        print(feat2[:10])