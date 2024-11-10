import time
import torch
from torch.utils.data import Dataset
import pandas as pd
from multiprocessing import Pool, cpu_count


class SingleProcessDataset(Dataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using single process...")
        
        self.data = pd.read_csv(csv_file)
        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def read_csv_chunk(csv_file, start, chunksize):
    return pd.read_csv(csv_file, skiprows=start, nrows=chunksize)


class MultiProcessDataset(SingleProcessDataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using multi process...")

        ########### YOUR CODE HERE ############
        with open(csv_file, "r") as f:
            num_lines = sum(1 for _ in f) - 1

        chunksize = num_lines // cpu_count()

        with Pool(cpu_count()) as p:
            data_chunks = p.starmap(
                read_csv_chunk,
                [(csv_file, i * chunksize, chunksize) for i in range(cpu_count())],
            )

        self.data = pd.concat(data_chunks, ignore_index=True)
        self.features = torch.FloatTensor(self.data[["x1", "x2", "x3"]].values)
        self.labels = torch.LongTensor(self.data["label"].values)
        ########### END YOUR CODE  ############
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")
