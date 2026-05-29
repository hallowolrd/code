import argparse
import random
import numpy as np
import torch.utils.data
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from args import parse
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, Subset,random_split
import os
from torchvision.datasets import ImageFolder

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


class Data:
    def __init__(self,parse:argparse.ArgumentParser):
        self.args = parse.parse_args()
        self.data_save_path = self.args.data_save_path
        self.num_clients = self.args.num_clients
        self.data_name = self.args.data_name
        self.alpha = self.args.alpha
        self.train_dataset,self.test_dataset,self.num_classes = self.load_dataset()
        self.dataset = ConcatDataset([self.train_dataset, self.test_dataset])
        self.batch_size = self.args.batch_size
        self.min_datasize = self.args.min_datasize
        self.sample_ratio = self.args.sample_ratio
        self.train_ratio = self.args.train_ratio
        self.val_ratio = self.args.val_ratio
        self.test_ratio = self.args.test_ratio

    def load_dataset(self):
        data_dict = {
            "cifar10": (CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),10),
            "cifar100":(CIFAR100,(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761),100)
        }

        setting = data_dict[self.data_name]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=setting[1],std=setting[2])
        ])
        train_dataset = setting[0](root='./data', train=True, download=True, transform=transform)
        test_dataset = setting[0](root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, setting[3]

    def split_dataset_by_dirichlet(self):

        if not (0 <= self.sample_ratio <= 1):
            raise ValueError("sample_ratio must be between 0 and 1")

        dataset_size = len(self.dataset)

        subsets_list = []
        # ============ server data ===============
        sample_size = int(dataset_size * self.sample_ratio)
        assert sample_size * self.val_ratio > self.batch_size
        indices = np.random.choice(dataset_size, size=sample_size, replace=False)
        server_subset = Subset(self.dataset, indices)
        subsets_list.append(server_subset)
        # ============ client data ===============
        train_labels =  np.concatenate([np.array(self.train_dataset.targets),np.array(self.test_dataset.targets)], axis=0)
        n_classes = train_labels.max() + 1
        label_distribution = np.random.dirichlet([self.alpha] * self.num_clients, self.num_classes)
        class_idcs = [np.argwhere(train_labels == y).flatten()
                      for y in range(n_classes)]
        client_idcs = [[] for _ in range(self.num_clients)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs,(np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
                client_idcs[i] += [idcs]
        for idcs in client_idcs:
            subsets_list.append(Subset(dataset=self.dataset,indices=np.concatenate(idcs)))


        return subsets_list


    def split_data(self,subset:Dataset):
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1
        total_size = len(subset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(subset,[train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset

    def extract_data(self, subset: Subset):
        dataset = subset.dataset
        indices = subset.indices
        data = [dataset[idx] for idx in indices]
        return data



    def save_dataset(self):
        subsets_list = self.split_dataset_by_dirichlet()
        for i in range(self.num_clients + 1):
            train_dataset, val_dataset, test_dataset = self.split_data(subsets_list[i])
            train_data = self.extract_data(train_dataset)
            val_data = self.extract_data(val_dataset)
            test_data = self.extract_data(test_dataset)
            if i == 0:
                torch.save(train_data, self.data_save_path + "/server_train.pth")
                torch.save(val_data, self.data_save_path + "/server_val.pth")
                torch.save(test_data, self.data_save_path + "/server_test.pth")
            else:
                torch.save(train_data, self.data_save_path + f"/{i}_train.pth")
                torch.save(val_data, self.data_save_path + f"/{i}_val.pth")
                torch.save(test_data, self.data_save_path + f"/{i}_test.pth")

class TinyImageNetData:
    def __init__(self, parse: argparse.ArgumentParser):
        self.args = parse.parse_args()
        data_temp = self.args.data_path
        self.data_path = data_temp+'/tiny-imagenet-200'+'/tiny-imagenet-200'
        self.data_save_path = self.args.data_save_path
        self.num_clients = self.args.num_clients
        self.alpha = self.args.alpha
        self.batch_size = self.args.batch_size
        self.sample_ratio = self.args.sample_ratio
        self.train_ratio = self.args.train_ratio
        self.val_ratio = self.args.val_ratio
        self.test_ratio = self.args.test_ratio

        self.train_dataset, self.val_dataset, self.test_dataset, self.num_classes = self.load_dataset()
        self.dataset = self.train_dataset

    def load_dataset(self):


        if not os.path.exists(f'{self.data_path}/'):
            os.system(f'wget --directory-prefix {self.data_path}/ http://cs231n.stanford.edu/tiny-imagenet-200.zip')
            os.system(f'unzip {self.data_path}/tiny-imagenet-200.zip')
        else:
            print('rawdata already exists.\n')

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        ])

        train_dataset = ImageFolder(root=os.path.join(self.data_path, "train"), transform=transform)
        val_dataset = ImageFolder(root=os.path.join(self.data_path, "val"), transform=transform)
        test_dataset = ImageFolder(root=os.path.join(self.data_path, "test"), transform=transform)

        num_classes = len(train_dataset.classes)

        return train_dataset, val_dataset, test_dataset, num_classes

    def split_dataset_by_dirichlet(self, seed=42):
        if not (0 <= self.sample_ratio <= 1):
            raise ValueError("sample_ratio must be between 0 and 1")

        np.random.seed(seed)
        dataset_size = len(self.dataset)

        subsets_list = []
        # ============ server data ===============
        sample_size = int(dataset_size * self.sample_ratio)
        indices = np.random.choice(dataset_size, size=sample_size, replace=False)
        server_subset = Subset(self.dataset, indices)
        subsets_list.append(server_subset)

        # ============ client data ===============
        targets = np.array([sample[1] for sample in self.train_dataset.samples])
        label_distribution = np.random.dirichlet([self.alpha] * self.num_clients, self.num_classes)
        class_idcs = [np.argwhere(targets == y).flatten() for y in range(self.num_classes)]
        client_idcs = [[] for _ in range(self.num_clients)]

        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
                client_idcs[i] += [idcs]

        for idcs in client_idcs:
            subsets_list.append(Subset(dataset=self.dataset, indices=np.concatenate(idcs)))

        return subsets_list

    def split_data(self, subset: Dataset):
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1
        total_size = len(subset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        test_size = total_size - train_size - val_size
        return random_split(subset, [train_size, val_size, test_size])

    def extract_data(self, subset: Subset):
        dataset = subset.dataset
        indices = subset.indices
        return [dataset[idx] for idx in indices]

    def save_dataset(self):
        os.makedirs(self.data_save_path, exist_ok=True)
        subsets_list = self.split_dataset_by_dirichlet()
        for i in range(self.num_clients + 1):
            train_dataset, val_dataset, test_dataset = self.split_data(subsets_list[i])
            train_data = self.extract_data(train_dataset)
            val_data = self.extract_data(val_dataset)
            test_data = self.extract_data(test_dataset)

            if i == 0:
                torch.save(train_data, os.path.join(self.data_save_path, "server_train.pth"))
                torch.save(val_data, os.path.join(self.data_save_path, "server_val.pth"))
                torch.save(test_data, os.path.join(self.data_save_path, "server_test.pth"))
            else:
                torch.save(train_data, os.path.join(self.data_save_path, f"{i}_train.pth"))
                torch.save(val_data, os.path.join(self.data_save_path, f"{i}_val.pth"))
                torch.save(test_data, os.path.join(self.data_save_path, f"{i}_test.pth"))

if __name__ == '__main__':
    args = parse.parse_args()
    dataname = args.data_name
    if dataname == "cifar10":
        Data(parse=parse).save_dataset()
    elif dataname == "cifar100":
        Data(parse=parse).save_dataset()
    elif dataname == "tinyimagenet":
        TinyImageNetData(parse=parse).save_dataset()
