import argparse
from sklearn.cluster import KMeans
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import sys
sys.path.append('./')
from sklearn.linear_model import LinearRegression
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.model_ours import DINOHead, SupConLoss_2,info_nce_logits, info_nce_logits_tail, SupConLoss,DistillLoss4,DistillLoss4_var, DistillLoss,DistillLoss4_weight, ContrastiveLearningViewGenerator, get_params_groups
from model.moco_tailness_cluster_forsup2 import MoCo
import matplotlib.pyplot as plt
from copy import deepcopy

    
def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    from util.cluster_and_log_utils import set_args_mmf
    set_args_mmf(args, train_loader)
    var=torch.zeros(args.num_labeled_classes + args.num_unlabeled_classes).cuda()
    # For lt evaluation

    
    
    queue = [[] for i in range(args.num_labeled_classes + args.num_unlabeled_classes)]
    
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
    
    best_test_acc_all_cl = -1
    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    

    # inductive
    best_test_acc_lab = 0
    best_test_acc_ubl = 0
    best_test_acc_all = 0
    best_train_acc_lab = 0
    best_train_acc_ubl = 0
    best_train_acc_all = 0
    
   
    
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
         
        student.train()
        # proj_moco.train()
       
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # contrastive_logits, contrastive_labels, tailness, student_out,student_feat = student(images[0].cuda(non_blocking=True),images[1].cuda(non_blocking=True), args)
                # student_proj, student_out, _ = student(images)
                student_proj, student_out, student_feat = student(torch.cat(images, dim=0).cuda(non_blocking=True))
                teacher_out = student_out.detach()    
                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out,epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
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
        
        # var = [np.var(i) for i in queue]
        var = [np.std(i) for i in queue]
        var = torch.tensor(var).cuda()
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        
        with torch.no_grad():
            all_acc_test_cl, old_acc_test_cl, new_acc_test_cl, acc_list_cl, bacc_list_cl, cl_ind_map = test(
                student,
                test_loader = test_loader,
                epoch = epoch,
                save_name='Test ACC',
                args=args)
            
        args.logger.info(
                'Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(all_acc_test_cl,
                                                                                    old_acc_test_cl,
                                                                                    new_acc_test_cl))
        
        
        

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if all_acc_test_cl > best_test_acc_all_cl:
        
            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            best_test_acc_new_cl = new_acc_test_cl
            best_test_acc_old_cl = old_acc_test_cl
            best_test_acc_all_cl = all_acc_test_cl
            
            # best_train_acc_new_cl = new_acc_test_cl_t
            # best_train_acc_old_cl = old_acc_test_cl_t
            # best_train_acc_all_cl = all_acc_test_cl_t


        args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on train set: All: {best_train_acc_all_cl:.1f} Old: {best_train_acc_old_cl:.1f} New: {best_train_acc_new_cl:.1f} ')
        args.logger.info(
        f'Metrics with best model on test set: All: {best_test_acc_all_cl:.1f} Old: {best_test_acc_old_cl:.1f} New: {best_test_acc_new_cl:.1f} ')
        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        # if all_acc_test > best_test_acc_all:
        # # if all_acc > best_train_acc_all:
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     best_test_acc_ubl = new_acc_test
        #     best_test_acc_all = all_acc_test
        #     best_test_acc_head_lab = head_old_acc_test
        #     best_test_acc_med_lab = med_old_acc_test
        #     best_test_acc_few_lab = few_old_acc_test
        #     best_test_acc_head_ubl = head_new_acc_test
        #     best_test_acc_med_ubl =  med_new_acc_test
        #     best_test_acc_few_ubl =  few_new_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc

        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info('Metrics with best model on test set: All {:.4f} | Old {:.4f} | New {:.4f} | Head_old {:.4f} | Med_old {:.4f} | Few_old {:.4f} | Head_new {:.4f} | Med_new {:.4f} | Few_new  {:.4f}'.format(best_test_acc_all, best_test_acc_lab,best_test_acc_ubl, best_test_acc_head_lab, best_test_acc_med_lab, best_test_acc_few_lab, best_test_acc_head_ubl,  best_test_acc_med_ubl,  best_test_acc_few_ubl))
        
        


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        # images = images.cuda(non_blocking=True)
        with torch.no_grad():
            # _, _, _, logits , _ = model(images.cuda(non_blocking=True),None,_,args,train=False)
            _, logits, _ = model(images.cuda(non_blocking=True))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            # mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
            #                              else False for x in label]))
            mask = np.append(mask, np.array([True if x.item() in args.train_classes #for cub,scar
                                         else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    all_acc, old_acc, new_acc, acc_list, bacc_list, ind_map = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=test_loader)

    return all_acc, old_acc, new_acc, acc_list, bacc_list, ind_map




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits',default=False)

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
    parser.add_argument('--imbalance_rate', type=float, default=1)
    parser.add_argument('--split_train_val', type=bool,default=False)
    parser.add_argument('--imb_ratio', type=float, default=1)
    parser.add_argument('--rev', type=str, default='consis')
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
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
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

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
    
    ema_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        ema_backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in ema_backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in ema_backbone.named_parameters():
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

    # train_all_test_trans.unlabelled_dataset.transform = test_transform
    train_all_test_trans.unlabelled_dataset.datasets[0].transform = test_transform
    train_all_test_trans.unlabelled_dataset.datasets[1].transform = test_transform
    train_unlabelled_test_trans.datasets[0].transform = test_transform
    train_unlabelled_test_trans.datasets[1].transform = test_transform
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
    
    train_loader_init =  DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              drop_last=False, pin_memory=True)
    train_unlabelled_test_trans_loader = DataLoader(train_unlabelled_test_trans, num_workers=args.num_workers, # 이게 transductive testset
                                      batch_size=args.batch_size, shuffle=False, pin_memory=False)
     
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    
    
    all_feats = []
    
    backbone.eval()
    
    backbone = backbone.to(device)
    
    for batch_idx, (images, label, _,_) in enumerate(tqdm(train_loader)):
              
        images = torch.cat(images, dim=0).cuda()
        
        feats = backbone(images)   
           
        feats = feats.to(device)

        feats = torch.nn.functional.normalize(feats, dim=-1)
        
        feats = feats.detach()
        
        all_feats.append(feats.cpu().numpy())
    
        
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
   
    centers_classifier = kmeans.cluster_centers_
   
    projector = DINOHead(args.feat_dim, args.mlp_out_dim, centers_classifier, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)
    
    # centers_proj = torch.randn((500,256))/math.sqrt(500) 
    # proj_moco = MoCo(base_encoder=backbone, ema_encoder=ema_backbone, args=args, centers_classifier = centers_classifier)
    # checkpoint = torch.load("/media/diml-03/shpark_gcd/shpark/GCD/simgcd/dev_outputs/Experiments_Memory/cifar100lt_moco_4096wSupCon_rere_(25.07.2023_|_26.943)/checkpoints/model_best.pt")
    # proj_moco.load_state_dict(checkpoint['model'])
    # model = proj_moco.to(device)
    
    all_feats = []
    
    model.eval()
    
    for batch_idx, (images, label, _,_) in enumerate(tqdm(train_loader)):
              
        images = torch.cat(images, dim=0).cuda()
        
        feats, _ ,_= model(images)   
           
        feats = feats.to(device)

        feats = torch.nn.functional.normalize(feats, dim=-1)
        
        feats = feats.detach()
        
        all_feats.append(feats.cpu().numpy())
    
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
   
    model.cluster_center=nn.functional.normalize(nn.Parameter(torch.tensor(kmeans.cluster_centers_)).cuda()).detach()
    
        
    # ----------------------
    # TRAIN
    # ----------------------
    train(model,train_loader, test_loader_labelled, test_loader_unlabelled, args)
