import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import tqdm


    
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, centers, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        last_layer = nn.Linear(in_dim, out_dim, bias=False)
        
        last_layer.weight = nn.Parameter(torch.tensor(centers).cuda())
        self.last_layer = nn.utils.weight_norm(last_layer)
        
        # self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
         
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x): #in x : 256,768
        
        # x_proj = nn.functional.normalize(self.mlp(x),dim=1)
          
        x_proj = self.mlp(x)
        
        x = nn.functional.normalize(x, dim=-1, p=2)
        
        # x = x.detach()
        
        logits = self.last_layer(x)
        
        return x_proj, logits
    
    
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, ema_encoder, args, centers_classifier, dim=256, K=4096, m=0.999, T=0.07, mlp=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = ema_encoder
        base_projector = DINOHead(args.feat_dim, args.mlp_out_dim, centers_classifier, nlayers=args.num_mlp_layers)
        ema_projector = DINOHead(args.feat_dim, args.mlp_out_dim, centers_classifier, nlayers=args.num_mlp_layers)
        if mlp:  # hack: brute-force replacement
           
            self.encoder_q = nn.Sequential(self.encoder_q,base_projector).cuda()
            self.encoder_k = nn.Sequential(self.encoder_k,ema_projector).cuda()
                
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda()
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.register_buffer("label_queue", torch.zeros(self.K).long())
        
        # self.cluster_center = nn.functional.normalize(nn.Parameter(torch.tensor(centers_proj)).cuda()).detach()
        
        self.cluster_center = None
        
        # self.register_buffer("cluster_center", torch.zeros(1, dtype=torch.long))
        
        # For Ablation
        
        self.register_buffer("variance_pseudo_epoch0", torch.randn(args.mlp_out_dim))
        self.register_buffer("variance_pseudo_epoch30", torch.randn(args.mlp_out_dim))
        self.register_buffer("variance_pseudo_epoch100", torch.randn(args.mlp_out_dim))
        self.register_buffer("variance_pseudo_epoch200", torch.randn(args.mlp_out_dim))
        
        self.register_buffer("variance_gt_epoch0", torch.randn(args.mlp_out_dim))
        self.register_buffer("variance_gt_epoch30", torch.randn(args.mlp_out_dim))
        self.register_buffer("variance_gt_epoch100", torch.randn(args.mlp_out_dim))
        self.register_buffer("variance_gt_epoch200", torch.randn(args.mlp_out_dim))
        
        self.register_buffer("tailness_epoch0_update", torch.randn(args.mlp_out_dim))
        self.register_buffer("tailness_epoch30_update", torch.randn(args.mlp_out_dim))
        self.register_buffer("tailness_epoch100_update", torch.randn(args.mlp_out_dim))
        self.register_buffer("tailness_epoch200_update", torch.randn(args.mlp_out_dim))

        self.register_buffer("tailness_epoch0", torch.randn(args.mlp_out_dim))
        self.register_buffer("tailness_epoch30", torch.randn(args.mlp_out_dim))
        self.register_buffer("tailness_epoch100", torch.randn(args.mlp_out_dim))
        self.register_buffer("tailness_epoch200", torch.randn(args.mlp_out_dim))

        
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, class_labels, args, train=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        
        # compute query features
        q_proj,q_out = self.encoder_q(im_q)  # queries: NxC
        
        q_proj = nn.functional.normalize(q_proj, dim=-1)
        
        if train==False:
            return None, None, None, q_out , q_proj
        
        # q = nn.functional.normalize(q, dim=1)
        q_proj2,q_out2 = self.encoder_q(im_k)
        
        q_proj2 = nn.functional.normalize(q_proj2, dim=-1)
        
        # compute key features
        
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_proj,k_out = self.encoder_k(im_k)  # keys: NxC
            
            k_proj = nn.functional.normalize(k_proj, dim=-1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q_proj, k_proj]).unsqueeze(-1) # 128,1
        # negative logits: NxK
        # l_neg = torch.einsum("nc,ck->nk", [q_proj, self.queue.clone().detach()]) # 128,65536
        l_neg = torch.einsum("nc,ck->nk", [q_proj, self.queue.clone().detach()]) # 128,65536
        
        # logits: Nx(1+K)
        
        logits = torch.cat([l_pos, l_neg], dim=1) # 128,65537
        
        assign_matrix = torch.matmul(k_proj,self.cluster_center.T)
        
        instance_assign = torch.argmax(assign_matrix,dim=-1)
        
        similarity_matrix = torch.matmul(self.cluster_center, self.queue.clone().detach()) #100,65536
        
        density_estimation = torch.zeros(similarity_matrix.shape[0]).detach() # 100,
        
        ## weighted knn
        # weights = similarity_matrix*0.1
        
        for i in range(similarity_matrix.shape[0]):
            # density_estimation[i] =  (torch.sort(similarity_matrix[i],descending=True)[0][:10] * (1/55)*torch.arange(10,0,-1).reshape(-1,1).cuda()).sum().clone().detach()
            density_estimation[i] =  (torch.sort(similarity_matrix[i],descending=True)[0][:15] * (1/120)*torch.arange(15,0,-1).reshape(-1,1).cuda()).sum().clone().detach()
                                     
            # density_estimation[i] /= 15#(1/3) * density_estimation[i]
            
            # for j in range(15): ## 15 = k in k-nn
            #     density_estimation[i] += (torch.sort(weights[i], descending=True)[0][j].clone().detach() * torch.sort(similarity_matrix[i],descending=True)[0][j].clone().detach()).cpu().sum().clone().detach()
            # density_estimation[i] /= torch.sort(weights[i], descending=True)[0][:15].cpu().sum().clone().detach()
        
        # apply temperature
        
        tau_min = density_estimation.min()
        
        tau_max = density_estimation.max()
        
        if False:
            
            min_tau = 0.07
            
            max_tau = 0.07
            
        else:
            
            min_tau = 0.05 # orig=0.05
            
            max_tau = 1 # orig = 1
        
        tau = (density_estimation-tau_min)/(tau_max-tau_min)
        
        tau = tau*(max_tau-min_tau)+min_tau
        
        instance_wise_tau = tau[instance_assign].cuda().detach()
        
        sup_logits = logits 
        
        logits = logits/instance_wise_tau.unsqueeze(1).expand(instance_wise_tau.shape[0],logits.shape[1])
        
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # dequeue and enqueue
        
        
        
        batch_size = labels.shape[0]
        # one-hot target from augmented image
        positive_target = torch.ones((batch_size, 1)).cuda()
        # find same label images from label queue
        # for the query with -1, all 
        targets = ((class_labels[:, None] == self.label_queue[None, :]) & (class_labels[:, None] != -1)).float().cuda()
        targets = torch.cat([positive_target, targets], dim=1)
        # dequeue and enqueue
        
        self._dequeue_and_enqueue(k_proj, class_labels)
        
        _, gathering_indices = torch.topk(assign_matrix, 1, dim=1)
        _, updating_indices = torch.topk(assign_matrix, 1, dim=0)
        query_update = self.get_update_query(self.cluster_center, gathering_indices, updating_indices,assign_matrix, k_proj)    
        self.cluster_center = nn.functional.normalize(0.1*query_update+0.9*self.cluster_center,dim=1).detach()
        return logits, labels, density_estimation[instance_assign] , torch.cat((q_out,q_out2),dim=0), torch.cat((q_proj,q_proj2)), sup_logits, targets
    
    def get_update_query(self, mem, max_indices, update_indices, score, query):
          
        m, d = mem.size()
        if True:
            query_update = torch.zeros((m,d)).cuda()
            
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                #ex = update_indices[0][i]
                if a != 0:
                    #random_idx = torch.randperm(a)[0]
                    #idx = idx[idx != ex]
#                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                    query_update[i] = torch.mean(query[idx].squeeze(1),dim=0)
                    #random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
                else:
                    query_update[i] = 0 
                    #random_update[i] = 0
        
       
            return query_update 
    
    

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

