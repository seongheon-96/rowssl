import torchvision
import numpy as np

import os

from copy import deepcopy
from data.data_utils import subsample_instances
from config import imagenet_root
import torch
import math
from PIL import Image

class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

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
        
        
def train_split(labels, n_labeled_per_class, n_unlabeled_per_class):
            labels = np.array(labels)
            train_labeled_idxs = []
            train_unlabeled_idxs = []
            for i in range(100):
                idxs = np.where(labels == i)[0]
                train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
                # train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
                train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
            return train_labeled_idxs, train_unlabeled_idxs

def subsample_dataset(dataset, idxs):


    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(100)), args=None):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

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


def get_imagenet_100_datasets(train_transform, test_transform, train_classes=range(50),
                           prop_train_labels=0.5, split_train_val=False, seed=0, args=None, imb=1, rev='consis'):

    np.random.seed(seed)

    if rev == 'consis':
            l_samples = make_imb_data(650, len(train_classes), imb, 1, 0)
            l_samples += (100 - len(train_classes))*[0]
            u_samples_known = make_imb_data(650, len(train_classes) , imb, 0, rev)
            u_samples_unknown = make_imb_data(1300, 100 - len(train_classes) , imb, 0, rev)
            u_samples = u_samples_known + u_samples_unknown
            
            
    elif rev == 'reverse':
        
            l_samples = make_imb_data(650, len(train_classes), imb, 1, 0)
            l_samples += (100 - len(train_classes))*[0]
            u_samples_known = make_imb_data(650, len(train_classes) , imb, 0, 1)
            u_samples_unknown = make_imb_data(1300, 100 - len(train_classes) , imb, 0, 1)
            u_samples = u_samples_known + u_samples_unknown 
            
    elif rev == 'uniform':
            l_samples = make_imb_data(250, len(train_classes), imb, 1, 0)
            l_samples += (100 - len(train_classes))*[0]
            u_samples_known = make_imb_data(250, len(train_classes) , 1, 0, rev)
            u_samples_unknown = make_imb_data(500, 100 - len(train_classes) , 1, 0, rev) # 이 부분 수정해야할듯
            u_samples = u_samples_known + u_samples_unknown 
            
    whole_training_set = ImageNetBase(root=os.path.join(imagenet_root, 'train'), transform=train_transform)
    train_labeled_idxs, train_unlabeled_idxs = train_split(whole_training_set.targets, l_samples, u_samples)

    train_dataset_labelled =  subsample_dataset(deepcopy(whole_training_set), train_labeled_idxs)
    
    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform
     
    # Get unlabelled data
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), train_unlabeled_idxs)
    
    lt_unlabeled_known_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=train_classes)
    subsample_indices = subsample_instances(lt_unlabeled_known_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_known_dataset = subsample_dataset(lt_unlabeled_known_dataset, subsample_indices) #3548
    
    lt_unlabeled_unknown_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=range(50,100))
    subsample_indices = subsample_instances(lt_unlabeled_unknown_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_unknown_dataset = subsample_dataset(lt_unlabeled_unknown_dataset, subsample_indices) # 16486
    
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    
    
    test_dataset = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=test_transform)

    # Transform dict
    unlabelled_classes = list(set(whole_training_set.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    
    
    # Get test set for all classes
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