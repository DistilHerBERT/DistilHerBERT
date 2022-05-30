from torch.utils.data import Dataset




class Zeros(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        data = torch.zeros(100) * 1.0
        data[idx] = 1
        data = data.reshape((1, 10, 10))

        if self.transform:
            data = self.transform(data)

        return data, i
