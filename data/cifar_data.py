# -*- coding: utf-8 -*-
import os
import sys
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

class CIFARDataset(torch.utils.data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, examples, transform=None, target_transform=None):
        self.examples = examples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.examples[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.examples)


class CIFAR10():
    """`CIFAR10` data reader.
    """

    base_folder = 'cifar-10-batches-py'
    filename = "cifar-10-python.tar.gz"
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
    }

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data, self.targets = list(), list()
        self.data_dict = {i: list() for i in range(10)}

        # now load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                labels = None
                if 'labels' in entry:
                    labels = entry['labels']
                else:
                    labels = entry['fine_labels']
                for k, v in zip(entry['data'], labels):
                    self.data.append(k)
                    self.targets.append(v)
                    self.data_dict[v].append(k)

        def convert(v):
            """convert to HWC"""
            v = np.vstack(v).reshape(-1, 3, 32, 32)
            v = v.transpose((0, 2, 3, 1))
            return v

        self.data = convert(self.data)
        for k, v in self.data_dict.items():
            self.data_dict[k] = convert(v)

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    @property
    def examples(self):
        return list(zip(self.data, self.targets))

    @property
    def examples_dict(self):
        return self.data_dict


class CifarData():
    """
    CIFAR10 allocator.
    """
    def __init__(self, client_num, sample_rate=-1, data_sharing=False, fine='CIFAR10'):
        if fine == 'CIFAR10':
            data_dir = 'dataset/cifar10'
        else:
            raise ValueError('Invalid dataset choice.')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                          std=[0.247, 0.243, 0.262])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_reader = CIFAR10(data_dir, train=True)
        test_reader = CIFAR10(data_dir, train=False)
        self.train_dict = train_reader.examples_dict
        total_train = train_reader.examples
        self.test_dict = test_reader.examples_dict
        total_test = test_reader.examples

        self.data_sharing = data_sharing
        if not 0 < sample_rate <= 1:
            sample_rate = 1
        np.random.shuffle(total_train)
        np.random.shuffle(total_test)
        self.share_train = total_train[:int(sample_rate * len(total_train))]
        self.share_test = total_test[:int(sample_rate * len(total_test))]

        self.num_local_train = len(total_train) // client_num
        self.num_local_test = len(total_test) // client_num

    def create_dataset_for_center(self, batch_size, num_workers):
        _train_set = CIFARDataset(self.share_train, self.train_transform)
        _test_set = CIFARDataset(self.share_test, self.test_transform)
        train_loader = DataLoader(_train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(_test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader, len(_train_set)

    def create_dataset_for_client(self, distribution, batch_size, num_workers, subset=tuple(range(10))):
        """
        subset: construct local data set with certain label(s).
        distribution: the distribution (of label space) to construct local data set.
        """
        distribution = np.asarray(distribution) / np.sum(distribution)

        def sample_data(data_dict, local_num):
            local_data = list()
            for i, p in enumerate(distribution):
                snum = int(local_num * p)
                indices = np.random.choice(len(data_dict[i]), snum, replace=False)
                local_data.extend([(k, i) for k in data_dict[i][indices]])
            return local_data

        local_train, local_test = list(), list()
        if len(subset) < 10:
            for i in subset:
                local_train.extend([(k, i) for k in self.train_dict[i]])
                local_test.extend([(k, i) for k in self.test_dict[i]])
        else:
            local_train = sample_data(self.train_dict, self.num_local_train)
            local_test = sample_data(self.test_dict, self.num_local_test)

        if self.data_sharing:
            local_train.extend(self.share_train)
            local_test.extend(self.share_test)

        _train_set = CIFARDataset(local_train, self.train_transform)
        _test_set = CIFARDataset(local_test, self.test_transform)
        train_loader = DataLoader(_train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(_test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader, len(local_train)


if __name__ == '__main__':
    pass
