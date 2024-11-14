import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import cub_root
import torch
import math
from PIL import Image

class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0): # [100,50,10,1] 등 클래스 별 샘플수가 롱테일되도록 개수 정해주는 용도
            mu = np.power(1/gamma, 1/(class_num - 1))
            class_num_list = []
            for i in range(class_num):
                if i == (class_num - 1):
                    class_num_list.append(int(max_num / gamma))
                else:
                    class_num_list.append(int(max_num * np.power(mu, i)))

            if flag == 0 and flag_LT == 1:
                class_num_list = list(reversed(class_num_list))
            print(class_num_list)
            return list(class_num_list)
        
        
def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, train_classes): 
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    unlabeled_classes = list(set(np.arange(200)) - set(train_classes))
    
    for i in range(200):
        idxs = np.where(labels == i)[0]
        
        if i in train_classes:
            train_labeled_idxs.extend(idxs[:n_labeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()]])
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()]:n_labeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()] + n_unlabeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()]])
        else:
            train_unlabeled_idxs.extend(idxs[:n_unlabeled_per_class[100 + torch.where((torch.tensor(unlabeled_classes))==i)[0].item()]])
    
    return train_labeled_idxs, train_unlabeled_idxs


def get_cub_datasets(train_transform, test_transform, train_classes=range(100), prop_train_labels=0.8,
                    split_train_val=False, seed=0, args=None,imb=1, rev='consis'):

    np.random.seed(seed)
    
    if rev == 'consis':
            l_samples = make_imb_data(15, len(train_classes), imb, 1, 0)
            l_samples += (200 - len(train_classes))*[0]
            u_samples_known = make_imb_data(15, len(train_classes) , imb, 0, rev)
            u_samples_unknown = make_imb_data(30, 200 - len(train_classes) , imb, 0, rev)
            u_samples = u_samples_known + u_samples_unknown
            
            
    elif rev == 'reverse':
            l_samples = make_imb_data(15, len(train_classes), imb, 1, 0)
            l_samples += (200 - len(train_classes))*[0]
            u_samples_known = make_imb_data(15, len(train_classes) , imb, 0, 1)
            u_samples_unknown = make_imb_data(30, 200 - len(train_classes) , imb, 0, 1)
            u_samples = u_samples_known + u_samples_unknown 
            

    # Init entire training set
    whole_training_set = CustomCub2011(root=cub_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    
    train_labeled_idxs, train_unlabeled_idxs = train_split(whole_training_set.data['target'].values-1, l_samples, u_samples, train_classes)

    train_dataset_labelled =  subsample_dataset(deepcopy(whole_training_set), train_labeled_idxs)
    
    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), train_unlabeled_idxs)
    
    # Get test set for all classes
    test_dataset = CustomCub2011(root=cub_root, transform=test_transform, train=False)    

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

if __name__ == '__main__':

    x = get_cub_datasets(None, None, split_train_val=False,
                         train_classes=range(100), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')