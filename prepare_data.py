import os
import argparse
import pandas as pd
from lib.utils import prepare_data

def prepare(root='NFBS_Dataset', out_path="NFBS_Dataset_meta.csv", out_dir='data'):
    # create empty dataframe
    df = pd.DataFrame(columns=["skull", "brain", "mask"])

    # iteratively append respective paths under each column
    for folders in os.listdir(root):
        files = [os.path.join(root, folders, file) for file in os.listdir(os.path.join(root, folders))]
        df = df.append({"skull": files[0], "brain": files[1], "mask": files[2]}, ignore_index=True)

    # save dataframe as csv
    df.to_csv(out_path, index=False)

    # rearrange the dataset
    prepare_data(out_path, out_dir)

    print(f"Dataset prepared at '{out_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root", required=True, type=str, help="NFBS Dataset's path")
    parser.add_argument("-o", "--out_path", required=False, default="NFBS_Dataset_meta.csv", type=str, help="Output csv file's name. Ex: data.csv")
    parser.add_argument("-d", "--out_dir", required=False, default="data", type=str, help="Output directory name to rearrange the dataset")

    args = parser.parse_args()

    prepare(args.root, args.out_path, args.out_dir)