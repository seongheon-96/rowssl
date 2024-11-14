from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np

from data.data_utils import subsample_instances
from config import cifar_10_root, cifar_100_root
from data.cifar_imb import cifar_imb
import math
from PIL import Image
import torch

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

def train_split10(labels, n_labeled_per_class, n_unlabeled_per_class):
            labels = np.array(labels)
            train_labeled_idxs = []
            train_unlabeled_idxs = []
            for i in range(10):
                idxs = np.where(labels == i)[0]
                train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
                # train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
                train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
            return train_labeled_idxs, train_unlabeled_idxs


                       
class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
   
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.1):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(math.ceil(val_split * len(cls_idxs)))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

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

def get_cifar_10_LT_datasets(train_transform, test_transform, train_classes=range(5),
                       prop_train_labels=0.8, split_train_val=False, seed=0, imb=1, rev='consis'):

    np.random.seed(seed)
    if rev == 'consis':
            l_samples = make_imb_data(2500, len(train_classes), imb, 1, 0)
            l_samples += (10 - len(train_classes))*[0]
            u_samples_known = make_imb_data(2500, len(train_classes) , imb, 0, rev)
            u_samples_unknown = make_imb_data(5000, 10 - len(train_classes) , imb, 0, rev)
            u_samples = u_samples_known + u_samples_unknown
            
            
    elif rev == 'reverse':
        
            l_samples = make_imb_data(2500, len(train_classes), imb, 1, 0)
            l_samples += (10 - len(train_classes))*[0]
            u_samples_known = make_imb_data(2500, len(train_classes) , imb, 0, 1)
            u_samples_known[-1] -= 1
            u_samples_known[-2] -= 1
            
            u_samples_unknown = make_imb_data(5000, 10 - len(train_classes) , imb, 0, 1)
            u_samples = u_samples_known + u_samples_unknown 
                      
            
    whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    train_labeled_idxs, train_unlabeled_idxs = train_split10(whole_training_set.targets, l_samples, u_samples)
    
    
    # Init entire training set
    # whole_training_set = cifar_imb(transform=train_transform,imbanlance_rate=imb,train=True,file_path=cifar_100_root, num_cls = 100)
    
    # del whole_training_set.x
    # whole_training_set = CustomCIFAR100_LT(root=cifar_100_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    # train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    # train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
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
    
    # Get unlabelled data
    # unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    # train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices))) #13958
    
    lt_unlabeled_known_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=train_classes)
    subsample_indices = subsample_instances(lt_unlabeled_known_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_known_dataset = subsample_dataset(lt_unlabeled_known_dataset, subsample_indices) #3548
    
    lt_unlabeled_unknown_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=range(5,10))
    subsample_indices = subsample_instances(lt_unlabeled_unknown_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_unknown_dataset = subsample_dataset(lt_unlabeled_unknown_dataset, subsample_indices) # 16486
    
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    
    
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

    
    
    # Get test set for all classes
    # test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    # test_dataset = Cifar100Imbanlance(imbanlance_rate=0.1,train=False,file_path=cifar_100_root)
    # del test_dataset.x
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


def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                       prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)
    
    # Init entire training set
    whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)

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
    
def get_cifar_100_LT_datasets(train_transform, test_transform, train_classes=range(50),
                       prop_train_labels=0.8, split_train_val=False, seed=0, imb=1, rev='consis'):

    np.random.seed(seed)
    if rev == 'consis':
            l_samples = make_imb_data(250, len(train_classes), imb, 1, 0)
            l_samples += (100 - len(train_classes))*[0]
            u_samples_known = make_imb_data(250, len(train_classes) , imb, 0, rev)
            u_samples_unknown = make_imb_data(500, 100 - len(train_classes) , imb, 0, rev)
            u_samples = u_samples_known + u_samples_unknown
            
            
    elif rev == 'reverse':
        
            l_samples = make_imb_data(250, len(train_classes), imb, 1, 0)
            l_samples += (100 - len(train_classes))*[0]
            u_samples_known = make_imb_data(250, len(train_classes) , imb, 0, 1)
            u_samples_unknown = make_imb_data(500, 100 - len(train_classes) , imb, 0, 1)
            u_samples = u_samples_known + u_samples_unknown 
            
    elif rev == 'uniform':
            l_samples = make_imb_data(250, len(train_classes), imb, 1, 0)
            l_samples += (100 - len(train_classes))*[0]
            u_samples_known = make_imb_data(250, len(train_classes) , 1, 0, rev)
            u_samples_unknown = make_imb_data(500, 100 - len(train_classes) , 1, 0, rev) # 이 부분 수정해야할듯
            u_samples = u_samples_known + u_samples_unknown 
            
            
        
        
    whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)
    train_labeled_idxs, train_unlabeled_idxs = train_split(whole_training_set.targets, l_samples, u_samples)
    
    
    # Init entire training set
    # whole_training_set = cifar_imb(transform=train_transform,imbanlance_rate=imb,train=True,file_path=cifar_100_root, num_cls = 100)
    
    # del whole_training_set.x
    # whole_training_set = CustomCIFAR100_LT(root=cifar_100_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    # train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    # train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
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
    
    # Get unlabelled data
    # unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    # train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices))) #13958
    
    lt_unlabeled_known_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=train_classes)
    subsample_indices = subsample_instances(lt_unlabeled_known_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_known_dataset = subsample_dataset(lt_unlabeled_known_dataset, subsample_indices) #3548
    
    lt_unlabeled_unknown_dataset = subsample_classes(deepcopy(train_dataset_unlabelled), include_classes=range(50,100))
    subsample_indices = subsample_instances(lt_unlabeled_unknown_dataset, prop_indices_to_subsample=1)
    lt_unlabeled_unknown_dataset = subsample_dataset(lt_unlabeled_unknown_dataset, subsample_indices) # 16486
    
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    
    
    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)

    # Transform dict
    unlabelled_classes = list(set(whole_training_set.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    
    
    # Get test set for all classes
    # test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    # test_dataset = Cifar100Imbanlance(imbanlance_rate=0.1,train=False,file_path=cifar_100_root)
    # del test_dataset.x
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

    x = get_cifar_100_datasets(None, None, split_train_val=False,
                         train_classes=range(80), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')