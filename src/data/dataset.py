import numpy as np
import torch
from torch.utils.data import Dataset

class InMemoryUnsupervisedDataset(Dataset):
    def __init__(self, source, transforms=None, t_max=None, is_squeeze=False, is_unsqueeze=False):
        self.transforms = transforms
        self.t_max = t_max
        self.is_squeeze = is_squeeze
        self.is_unsqueeze = is_unsqueeze

        if type(source) == type(str()):
            self.samples = np.load(source)

        elif type(source) == type(np.array([])):
            self.samples = source

        else:
            print(f"ERROR: UNKNOWN DATA TYPE {type(source)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = np.array(self.samples[index])

        if self.transforms is not None: 
            for transform in self.transforms:
                sample = transform(sample)

        #scaling
        sample = torch.from_numpy(
            (sample - sample.min())/(sample.max() - sample.min())
        ).to(torch.float32)
        if self.is_squeeze: sample = sample.squeeze()
        if self.is_unsqueeze: sample = sample.unsqueeze(axis=-3)
        if self.t_max is not None: sample = sample[..., :self.t_max]
            
        return sample

    def random_shuffled_iterator(self, batch_size: int):
        while True:
            inds = torch.randint(0, len(self.samples), [batch_size])
            yield torch.stack([self.__getitem__(idx) for idx in inds])

class InMemorySupervisedDataset(Dataset):
    def __init__(self, source=None, source_samples=None, source_labels=None, transforms=None, t_max=None, is_squeeze=False, is_unsqueeze=False):
        self.transforms = transforms
        self.t_max = t_max
        self.is_squeeze = is_squeeze
        self.is_unsqueeze = is_unsqueeze

        if source is not None:
            self.data = source
        else:
            samples = None
            labels = None
            if type(source_samples) == type(str()):
                samples = np.load(source_samples)
            elif type(source_samples) == type(np.array([])):
                samples = source_samples
            else:
                print(f"ERROR: UNKNOWN DATA TYPE {type(source_samples)}")

            if type(source_labels) == type(str()):
                labels = np.load(source_labels)
            elif type(source_labels) == type(np.array([])):
                labels = source_labels
            else:
                print(f"ERROR: UNKNOWN DATA TYPE {type(source_labels)}")

            assert len(samples) == len(labels), "Labels and samples don't have same length"
            
            self.data = list(zip(samples, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample, label = self.data[index]
        label = torch.tensor(label)
        sample = np.array(sample)

        if self.transforms is not None: 
            for transform in self.transforms:
                sample = transform(sample)

        #scaling
        sample = torch.from_numpy(
            (sample - sample.min())/(sample.max() - sample.min())
        ).to(torch.float32)
        if self.is_squeeze: sample = sample.squeeze()
        if self.is_unsqueeze: sample = sample.unsqueeze(axis=-3)
        if self.t_max is not None: sample = sample[..., :self.t_max]
            
        return sample, label