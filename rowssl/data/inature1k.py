import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random
import os
import json
from tqdm import tqdm
from data.data_utils import subsample_instances
from config import iNaturalist18_root
from copy import deepcopy
# modified from https://github.com/jiequancui/ResLT/blob/3f6b0ad95223f3afc9b4a4cc9d208149d1744538/Inat/datasets/inaturalist2018.py & https://github.com/kleinzcy/NCDLR/blob/main/data/inature.py


class iNaturalist18Dataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, root='/data2/kh12043/datasets/iNaturalist18'):
        self.samples = []
        self.targets = []
        self.transform = transforms
        with open(txt) as f:
            for line in f:
                self.samples.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.transform = transform
        self.target_transform = target_transform
        
        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.samples[index]
        label = self.targets[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample, label, self.uq_idxs[index]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    # dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    # dataset.targets = [int(x) for x in dataset.targets]

    return dataset


def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # Have a balanced test set
        v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_inaturelist18_datasets(train_transform,
                               test_transform,
                               train_classes=range(500),
                               seed=0, prop_train_labels=0.5,
                               split_train_val=False, imb=1, rev='consis', args=None):
    
    num_labeled_classes = len(train_classes)
    num_unlabeled_classes = len(train_classes)
    
    num_classes = 8142
    
    train_txt = os.path.join('/data2/kh12043/datasets/iNaturalist18', "iNaturalist18_1k_train.txt")
    val_txt = os.path.join('/data2/kh12043/datasets/iNaturalist18', "iNaturalist18_1k_val.txt")

    np.random.seed(0)
    all_classes = np.random.choice(range(num_classes), size=len(train_classes)*2, replace=False)
    
    train_classes = all_classes[:num_labeled_classes]
    val_classes = all_classes[num_unlabeled_classes:] ## novel class

    target_xform_dict = {}
    for i, k in enumerate(all_classes):
        target_xform_dict[k] = i
    
    # Init entire training set
    train_dataset = iNaturalist18Dataset(train_txt, transform=train_transform)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    # TODO: Subsampling unlabelled set in uniform random fashion from training data, will contain many instances of dominant class
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    if split_train_val:

        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

    else:

        train_dataset_labelled_split, val_dataset_labelled_split = None, None


    
    # Get unlabelled data
    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))
    
    # Get test dataset
    test_dataset = iNaturalist18Dataset(val_txt, transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=all_classes)
    

    # Transform dict
    unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == "__main__":
   
    np.random.seed(0)
    train_classes = np.random.choice(range(8192,), size=(1000), replace=False)

    x = get_inaturelist18_datasets(None, None, train_classes=train_classes,
                               prop_train_labels=0.8)


    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
    print('Printing number of labelled classes...')
    print(len(set(x['train_labelled'].targets)))
    print('Printing total number of classes...')
    print(len(set(x['train_unlabelled'].targets)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')