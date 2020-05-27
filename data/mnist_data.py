# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class MNIST(torch.utils.data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        examples (list: [(img, target)...] )
    """
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

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
        target = int(target)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.examples)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}


class MNISTData():
    """
    MNIST allocator.
    """
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, client_num, sample_rate=-1, data_sharing=False):
        data_dir = 'dataset/mnist'
        self.root = os.path.expanduser(data_dir)

        if not self._check_exists():
            print(self.root)
            raise RuntimeError('Dataset not found.')

        self.train_dict, total_train = self._read_file(self.training_file)
        self.test_dict, total_test = self._read_file(self.test_file)

        self.data_sharing = data_sharing
        if not 0 < sample_rate <= 1:
            sample_rate = 1
        np.random.shuffle(total_train)
        np.random.shuffle(total_test)
        self.share_train = total_train[:int(sample_rate * len(total_train))]
        self.share_test = total_test[:int(sample_rate * len(total_test))]

        self.num_local_train = len(total_train) // client_num
        self.num_local_test = len(total_test) // client_num

        normalize = transforms.Normalize(mean=[0.131], std=[0.308])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def create_dataset_for_center(self, batch_size, num_workers):
        _train_set = MNIST(self.share_train, self.train_transform)
        _test_set = MNIST(self.share_test, self.test_transform)
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

        _train_set = MNIST(local_train, self.train_transform)
        _test_set = MNIST(local_test, self.test_transform)
        train_loader = DataLoader(_train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(_test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader, len(local_train)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def _read_file(self, data_file):
        """
        return:
            data: (dict: {label: array[images, ...]} )
            total_data: (list [(image, label), ...] )
        """
        data = {i: [] for i in range(10)}
        total_data, total_targets = torch.load(os.path.join(self.processed_folder, data_file))
        total_data = [x.numpy() for x in total_data]
        for k, v in zip(total_data, total_targets):
            data[int(v)].append(k)
        for k, v in data.items():
            data[k] = np.asarray(v)
        return data, list(zip(total_data, total_targets))

if __name__ == "__main__":
    pass
