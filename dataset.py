import torch.utils.data as data
import torch
import h5py


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, file_path):
        super(TrainDatasetFromFolder, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('train/data')
        self.target = hf.get('train/label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.target[index])

    def __len__(self):
        return self.data.shape[0]

