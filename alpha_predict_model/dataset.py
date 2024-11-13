import torch
from torch.utils.data import DataLoader, Dataset


class DNN_logit_dataset(Dataset):
    def __init__(self, pickle_data, pickle_data_reference, target):
        self.combined_data = [
            torch.cat([pickle_data[i][0].squeeze(0), pickle_data_reference[i][0].squeeze(0)]).float().flatten()
            for i in range(len(pickle_data))
        ]
        self.target = torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, item):
        return {"input_tensor": self.combined_data[item], "labels": self.target[item]}


def create_data_loader(pickle_data, pickle_data_reference, df, batch_size, shuffle_=False):
    ds = DNN_logit_dataset(
        pickle_data=pickle_data,
        pickle_data_reference=pickle_data_reference,
        target=df.iloc[:, 20:37].to_numpy(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_)
