import argparse
from sklearn.cluster import KMeans as KMeans_
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from fast_pytorch_kmeans import KMeans
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.model_ours import DINOHead_gcd, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from model.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

def train(model , train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(model)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    
    from util.cluster_and_log_utils import set_args_mmf
    set_args_mmf(args, train_loader)
    
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    best_test_acc_all_cl = -1
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    sup_con_crit = SupConLoss()

    # inductive
    best_test_acc_lab = 0
    best_test_acc_ubl = 0
    best_test_acc_all = 0
    # transductive
    best_train_acc_lab = 0
    best_train_acc_ubl = 0 
    best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                features = model(images)

                # Pass features through projection head
                # features = projection_head(features)

                # L2-normalize features
                features = torch.nn.functional.normalize(features, dim=-1)

                # Choose which instances to run the contrastive loss on
                
                con_feats = features

                contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[mask_lab] for f in features.chunk(2)]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = class_labels[mask_lab]

                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

                # Total loss
                
                loss = (1 - 0.35 ) * contrastive_loss + 0.35 * sup_con_loss

                pstr = ''
                pstr += f'unsup_loss: {contrastive_loss.item():.4f} '
                pstr += f'supcon_loss: {sup_con_loss.item():.4f} '
                
            
            
                    
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                    loss.backward()
                    optimizer.step()
            else:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                    args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Testing on unlabelled examples in the training data...')
            
        with torch.no_grad():
                all_acc_test_cl_t, old_acc_test_cl_t, new_acc_test_cl_t, acc_list_cl_t, bacc_list_cl_t, cl_ind_map_t, kmeans = test_transductive(
                    model,
                    test_loader=train_loader,
                    epoch=epoch,
                    save_name='Transductive Test ACC',
                    args=args,
                    train_loader=train_loader)
        args.logger.info(
                    'Transductive Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(all_acc_test_cl_t,
                                                                                        old_acc_test_cl_t,
                                                                                        new_acc_test_cl_t))
            # with torch.no_grad():
            #     all_acc_test_cl_t, old_acc_test_cl_t, new_acc_test_cl_t, acc_list_cl_t, bacc_list_cl_t, cl_ind_map_t, kmeans = test_kmeans_semi(
            #         model,
            #         test_loader=train_loader,
            #         epoch=epoch,
            #         save_name='Transductive Test ACC',
            #         args=args,
            #         train_loader=train_loader)
            
            # args.logger.info(
            #         'Transductive Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(all_acc_test_cl_t,
            #                                                                             old_acc_test_cl_t,
            #                                                                             new_acc_test_cl_t))
        args.logger.info('Testing on disjoint test set...')
        with torch.no_grad():
                all_acc_test_cl, old_acc_test_cl, new_acc_test_cl, acc_list_cl, bacc_list_cl, cl_ind_map = test_inductive(
                    model,
                    test_loader,
                    kmeans,
                    epoch=epoch,
                    save_name='Test ACC',
                    args=args,
                    train_loader=train_loader)
            
        args.logger.info(
                    'Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(all_acc_test_cl,
                                                                                      old_acc_test_cl,
                                                                                      new_acc_test_cl))

        args.logger.info('Testing on disjoint test set w/o matching...')
        with torch.no_grad():
                all_acc_test_cl2, old_acc_test_cl2, new_acc_test_cl2,bacc_many_old, bacc_med_old, bacc_few_old, bacc_many_new, bacc_med_new, bacc_few_new = test_inductive_(
                    model,
                    test_loader,
                    kmeans,
                    epoch=epoch,
                    save_name='Test ACC',
                    ind = cl_ind_map_t,
                    args=args)
            
        args.logger.info(
                    'Test Accuracies w/o matching CL: All {:.1f} | Old {:.1f} | New {:.1f} | Many Old {:.1f} | Med Old {:.1f} | Few Old {:.1f} | Many New {:.1f} | Med New {:.1f} | Few New {:.1f}'.format(all_acc_test_cl2,
                                                                                      old_acc_test_cl2,
                                                                                      new_acc_test_cl2,bacc_many_old, bacc_med_old, bacc_few_old, bacc_many_new, bacc_med_new, bacc_few_new))
        
        # args.logger.info('Testing on disjoint test set like BaCon')
        # with torch.no_grad():
        #     all_acc_test_cl_tt, old_acc_test_cl_tt, new_acc_test_cl_tt, acc_list_cl_tt, bacc_list_cl_tt, cl_ind_map_tt = test_bacon(
        #         model,
        #         test_loader=test_loader,
        #         epoch=epoch,
        #         save_name='Bacon Test ACC',
        #         args=args)
        # args.logger.info(
        #         'Bacon Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(all_acc_test_cl_tt,
        #                                                                             old_acc_test_cl_tt,
        #                                                                             new_acc_test_cl_tt))
        
        # Step schedule
        exp_lr_scheduler.step()
    

    #     if all_acc_test_cl > best_test_acc_all_cl:

    #         best_test_acc_new_cl = new_acc_test_cl
    #         best_test_acc_old_cl = old_acc_test_cl
    #         best_test_acc_all_cl = all_acc_test_cl
            
    #         best_test_acc_new_cl2 = new_acc_test_cl2
    #         best_test_acc_old_cl2 = old_acc_test_cl2
    #         best_test_acc_all_cl2 = all_acc_test_cl2

    #         best_train_acc_new_cl = new_acc_test_cl_t
    #         best_train_acc_old_cl = old_acc_test_cl_t
    #         best_train_acc_all_cl = all_acc_test_cl_t

    #         save_dict = {
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'epoch': epoch + 1,
    #         }

    #         torch.save(save_dict, args.model_path)
    #         args.logger.info("model saved to {}.".format(args.model_path))

           

  
    # args.logger.info(f'Metrics with best model on train set: All: {best_train_acc_all_cl:.1f} Old: {best_train_acc_old_cl:.1f} New: {best_train_acc_new_cl:.1f} ')
    # args.logger.info(
    #     f'Metrics with best model on test set: All: {best_test_acc_all_cl:.1f} Old: {best_test_acc_old_cl:.1f} New: {best_test_acc_new_cl:.1f} ')
    # args.logger.info(
    #     f'Metrics with best model on test set w/o matching: All: {best_test_acc_all_cl2:.1f} Old: {best_test_acc_old_cl2:.1f} New: {best_test_acc_new_cl2:.1f} ')


def test_inductive(model, test_loader, kmeans_train, epoch, save_name, args, train_loader):
    model.eval()

    all_feats = []
    preds = []
    targets = np.array([])
    mask = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(test_loader):
        batch = batch[:3]
        (images, label, _) = batch
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model[0](images)  # follow GCD: clustering on normalized backbone feature
        
        feats = torch.nn.functional.normalize(feats, dim=-1)
     
        dist = torch.matmul(feats,torch.tensor(kmeans_train.centroids.T).cuda())
        preds.append(torch.argmax(dist,dim=1).cpu().numpy())
        
        # all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
        # mask = np.append(mask, np.array([True if x.item() in args.train_classes #for cub,scar, inat
        #                                 else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    # all_feats = np.concatenate(all_feats)
    # kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    # preds = kmeans.labels_
    # print('Done!')
    preds = np.concatenate(preds)
    all_acc, old_acc, new_acc, acc_list,bacc_list, ind_map = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=train_loader)

    return all_acc, old_acc, new_acc, acc_list, bacc_list, ind_map

def test_inductive_(model, test_loader, kmeans_train, epoch, save_name, ind, args):
    
    model.eval()

    all_feats = []
    preds = []
    targets = np.array([])
    mask = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(test_loader):
        batch = batch[:3]
        (images, label, _) = batch
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model[0](images)  # follow GCD: clustering on normalized backbone feature
        
        feats = torch.nn.functional.normalize(feats, dim=-1)
     
        dist = torch.matmul(feats,torch.tensor(kmeans_train.centroids.T).cuda())
        preds.append(torch.argmax(dist,dim=1).cpu().numpy())
        
        # all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # if args.dataset_name == 'cub':
            # mask = np.append(mask, np.array([True if x.item() in args.train_classes #for cub
            #                             else False for x in label]))
        # else:
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                    else False for x in label]))

    preds = np.concatenate(preds)
    # targets = np.concatenate(targets)
    
    mask = mask.astype(bool)
    targets = targets.astype(int)
    preds = preds.astype(int)
     
    all_classes_gt = set(targets)
    old_classes_gt = set(targets[mask])
    new_classes_gt = set(targets[~mask])
    
    
    D = max(preds.max(), targets.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        w[preds[i], targets[i]] += 1
        
    bacc_list = []
    bacc_list_targets=[args.k_cls_idx,args.uk_cls_idx, args.known_many_cls, args.known_median_cls, args.known_few_cls,
                                                      args.unknown_many_cls, args.unknown_median_cls, args.unknown_few_cls]
    for i in range(w.shape[0]):
        
        acc = w[ind[i], i]
        total_instances = sum(w[:, i])

        if total_instances != 0:
            acc /= total_instances
            bacc_list.append(acc * 100)
            
        else:
            bacc_list.append(0)
            
    bacc_all = sum(bacc_list)/len(bacc_list)
    
    bacc_old_list = []
    bacc_new_list = []

    bacc_many_old_list = []
    bacc_med_old_list = []
    bacc_few_old_list = []

    bacc_many_new_list = []
    bacc_med_new_list = []
    bacc_few_new_list = []
    
    for i in bacc_list_targets[0]: # For Old Classes
        bacc_old_list.append(bacc_list[i])
        
    for i in bacc_list_targets[1]: # For New Classes
        bacc_new_list.append(bacc_list[i])

    for i in bacc_list_targets[2]: # For Old Classes
        bacc_many_old_list.append(bacc_list[i])
    for i in bacc_list_targets[3]: # For Old Classes
        bacc_med_old_list.append(bacc_list[i])
    for i in bacc_list_targets[4]: # For Old Classes
        bacc_few_old_list.append(bacc_list[i])
    for i in bacc_list_targets[5]: # For Old Classes
        bacc_many_new_list.append(bacc_list[i])
    for i in bacc_list_targets[6]: # For Old Classes
        bacc_med_new_list.append(bacc_list[i])
    for i in bacc_list_targets[7]: # For Old Classes
        bacc_few_new_list.append(bacc_list[i])

    bacc_old = sum(bacc_old_list)/len(bacc_old_list)              
    bacc_new = sum(bacc_new_list)/len(bacc_new_list) 

    bacc_many_old = sum(bacc_many_old_list)/len(bacc_many_old_list)
    bacc_med_old = sum(bacc_med_old_list)/len(bacc_med_old_list)
    bacc_few_old = sum(bacc_few_old_list)/len(bacc_few_old_list)

    bacc_many_new = sum(bacc_many_new_list)/len(bacc_many_new_list)
    bacc_med_new = sum(bacc_med_new_list)/len(bacc_med_new_list)
    bacc_few_new = sum(bacc_few_new_list)/len(bacc_few_new_list)

    return bacc_all, bacc_old, bacc_new, bacc_many_old, bacc_med_old, bacc_few_old, bacc_many_new, bacc_med_new, bacc_few_new

def test_transductive(model, test_loader, epoch, save_name, args, train_loader):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    mask_lab = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(test_loader):
        # batch = batch[:]
        (images, label, _, mask_lab_) = batch
        mask_lab_ = mask_lab_[:, 0]
        
        images = images[0]
        images = images.cuda()
        
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model[0](images)  # follow GCD: clustering on normalized backbone feature

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # if args.dataset_name == 'cub':
        #     mask = np.append(mask, np.array([True if x.item() in args.train_classes #for cub
        #                                     else False for x in label]))
    # else: 
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                    else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())
        
    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    all_feats = torch.tensor(all_feats).cuda()
    
    kmeans = KMeans(n_clusters=int(1*(args.num_labeled_classes + args.num_unlabeled_classes)), verbose=1)
    labels = kmeans.fit_predict(all_feats)
    # kmeans = KMeans(n_clusters = args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    # preds = kmeans.labels_
    preds = np.array(labels.cpu())
    print('Done!')
    mask_lab = mask_lab.astype(bool)
    targets = targets[~mask_lab]
    preds = preds[~mask_lab]
    mask = mask[~mask_lab]
    
    all_acc, old_acc, new_acc, acc_list, bacc_list, ind_map = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=train_loader)

    return all_acc, old_acc, new_acc, acc_list, bacc_list, ind_map, kmeans

def test_bacon(model, test_loader, epoch, save_name, args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(test_loader):
        batch = batch[:3]
        (images, label, _) = batch
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model[0](images)  # follow GCD: clustering on normalized backbone feature

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans_(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    all_acc, old_acc, new_acc, acc_list, bacc_list,ind_map = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=test_loader)

    return all_acc, old_acc, new_acc, acc_list, bacc_list,ind_map

def test_kmeans_semi(model, test_loader, epoch, save_name, args, train_loader):
    model.eval()
    
    all_feats = []
    targets = np.array([])
    mask_cls = np.array([])
    mask_lab = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(test_loader):
        # batch = batch[:]
        (images, label, _, mask_lab_) = batch
        mask_lab_ = mask_lab_[:, 0]
        images = images[0]
        images = images.cuda()
        
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model[0](images)  # follow GCD: clustering on normalized backbone feature

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())
        
        
    # -----------------------
    # K-MEANS
    # -----------------------
    
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets

    print('Fitting Semi-Supervised K-Means...')
    
    K = args.num_labeled_classes + args.num_unlabeled_classes
    
    args.max_kmeans_iter = 200
    
    args.k_means_init = 10
    
    kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                           n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)
    
    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeans.fit_mix(u_feats, l_feats, l_targets)
    
    all_preds = kmeans.labels_.cpu().numpy()
    
    u_targets = u_targets.cpu().numpy()

    # -----------------------
    # EVALUATE
    # -----------------------
    # Get preds corresponding to unlabelled set
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)
    
    
    
    all_acc, old_acc, new_acc, acc_list, bacc_list, ind_map = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=train_loader)

    return all_acc, old_acc, new_acc, acc_list, bacc_list, ind_map, kmeans

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--imb_ratio', default=100, type=int)
    parser.add_argument('--split_train_val', type=bool,default=False)
    parser.add_argument('--rev', type=str, default='consis')
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    # if args.dataset_name == 'cifar_10':
    #     total_class = 10
    # elif args.dataset_name == 'herbarium_19':
    #     total_class = 683
    # else:
    #     total_class = 100
    #     args.labeled_classes = 50
    #     args.unlabeled_classes = range(args.labeled_classes, total_class)
    
    # if args.dataset_name == 'herbarium_19':
            
    #     herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')
    #     with open(herb_path_splits, 'rb') as handle:
    #         class_splits = pickle.load(handle)
    #     args.train_classes = class_splits['Old']
    #     args.unlabeled_classes = class_splits['New']
    #     args.labeled_classes = args.train_classes
    
    # if args.dataset_name != 'herbarium_19':
    #     args.train_classes = range(args.labeled_classes)
    #     args.unlabeled_classes = range(args.labeled_classes, total_class)
 
    
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    
    
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['Cifar10'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536
    

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
  
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_all_test_trans = deepcopy(train_dataset)
    train_labelled_test_trans = deepcopy(train_dataset.labelled_dataset)
    train_unlabelled_test_trans = deepcopy(train_dataset.unlabelled_dataset)

    train_all_test_trans.labelled_dataset.transform = test_transform

    train_all_test_trans.unlabelled_dataset.datasets[0].transform = test_transform
    train_all_test_trans.unlabelled_dataset.datasets[1].transform = test_transform
    train_unlabelled_test_trans.datasets[0].transform = test_transform
    train_unlabelled_test_trans.datasets[1].transform = test_transform
    # train_all_test_trans.unlabelled_dataset.transform = test_transform
    # train_unlabelled_test_trans.transform = test_transform
    train_labelled_test_trans.transform = test_transform

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False, pin_memory=False)
    
   
    train_unlabelled_test_trans_loader = DataLoader(train_unlabelled_test_trans, num_workers=args.num_workers, # 이게 transductive testset
                                      batch_size=args.batch_size, shuffle=False, pin_memory=False)
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead_gcd(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, train_unlabelled_test_trans_loader ,args)
