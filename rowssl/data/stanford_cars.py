import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import imagenet_root
import torch
import math
from PIL import Image

car_root = "/mnt/yids-01/kh12043/datasets/stanford_car"
meta_default_path = "/mnt/yids-01/kh12043/datasets/stanford_car/cars_{}.mat"
#         data_dir = os.path.join(data_dir, 'cars_train/') if train else os.path.join(data_dir, 'cars_test/')

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None, metas=meta_default_path):
        # data_dir2 = '/data2/kh12043/shpark_gcd/shpark/GCD/datasets/'
        data_dir = os.path.join(data_dir, 'cars_train/') if train else os.path.join(data_dir, 'cars_test/')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)

def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0):
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
    unlabeled_classes = list(set(np.arange(196)) - set(train_classes))
    for i in range(196):
        idxs = np.where(labels == i)[0]
        if i in train_classes:
            train_labeled_idxs.extend(idxs[:n_labeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()]])
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()]:n_labeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()] + n_unlabeled_per_class[torch.where((torch.tensor(train_classes))==i)[0].item()]])
        else:
            train_unlabeled_idxs.extend(idxs[:n_unlabeled_per_class[98 + torch.where((torch.tensor(unlabeled_classes))==i)[0].item()]])
    return train_labeled_idxs, train_unlabeled_idxs

def subsample_dataset(dataset, idxs):
    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_lt_dist(cls_num, img_max, imb_factor):
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return torch.tensor(img_num_per_cls)

def get_scars_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0,imb=1, rev='consis'):

    np.random.seed(seed)

    if rev == 'consis':
            l_samples = make_imb_data(20, len(train_classes), imb, 1, 0)
            l_samples += (196 - len(train_classes))*[0]
            u_samples_known = make_imb_data(20, len(train_classes) , imb, 0, rev)
            u_samples_unknown = make_imb_data(40, 196 - len(train_classes) , imb, 0, rev)
            u_samples = u_samples_known + u_samples_unknown
            
            
    elif rev == 'reverse':
        
            l_samples = make_imb_data(20, len(train_classes), imb, 1, 0)
            l_samples += (196 - len(train_classes))*[0]
            u_samples_known = make_imb_data(20, len(train_classes) , imb, 0, 1)
            u_samples_known[-1] -= 1
            u_samples_known[-2] -= 1
            u_samples_unknown = make_imb_data(40, 196 - len(train_classes) , imb, 0, 1)
            u_samples = u_samples_known + u_samples_unknown 

    # Init entire training set
    whole_training_set = CarsDataset(data_dir=car_root, transform=train_transform, metas=meta_default_path, train=True)
    print('known', sorted(train_classes))

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    # train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    train_labeled_idxs, train_unlabeled_idxs = train_split(list(np.array(whole_training_set.target)-1), l_samples, u_samples, train_classes)
    train_dataset_labelled =  subsample_dataset(deepcopy(whole_training_set), train_labeled_idxs)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    # unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    # train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), train_unlabeled_idxs)
    lt_unlabeled_known_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=train_classes)
    subsample_indices = subsample_instances(lt_unlabeled_known_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_known_dataset = subsample_dataset(lt_unlabeled_known_dataset, subsample_indices) #3548
    
    unlabelled_classes = list(set([x-1 for x in whole_training_set.target]) - set(train_classes))
    print('u', unlabelled_classes)
    # import pdb; pdb.set_trace()
    lt_unlabeled_unknown_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=unlabelled_classes)
    subsample_indices = subsample_instances(lt_unlabeled_unknown_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_unknown_dataset = subsample_dataset(lt_unlabeled_unknown_dataset, subsample_indices) # 16486
    
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    
    # Get test set for all classes
    test_dataset = CarsDataset(data_dir=car_root, transform=test_transform, metas=meta_default_path, train=False)

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

    x = get_scars_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')