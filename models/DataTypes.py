from torch.utils.data import Dataset

class EEGDataset(Dataset):

    def __init__(self, data_set, label_set, transform=None):
        
        self.transform = transform
        self.label_set = label_set
        self.data_set = data_set

    def __len__(self):
        return len(self.label_set)

    def __getitem__(self, idx):
        eeg_sample = self.data_set[idx,:,:]
        label = self.label_set[idx]

        if self.transform:
            eeg_sample = self.transform(eeg_sample)

        return eeg_sample, label