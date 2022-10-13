from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean
from typing import List

from torch.nn.modules import loss

import pointgroup_ops


class MultiTaskLoss(nn.Module):
    def __init__(self,
                 logger,
                 param_loss,
                 param_model):
        super().__init__()
        self.logger = logger
        self.ignore_label = param_loss.ignore_label
        self.supervise_instance_size = param_loss.supervise_instance_size
        self.joint_training_epoch = param_loss.joint_training_epoch
        self.semantic_dice = param_loss.semantic_dice
        self.semantic_class_num = param_model.classes

        # discriminative loss parm
        self.discriminative_feature_dim = 7
        self.delta_v = 0.1  # OccuSeg delta_v = 0.1
        self.delta_d = 1.5  # OccuSeg delta_d = 1.5
        self.param_var = 1.
        self.param_dist = 1.
        self.param_reg = 0.001
        self.device = 'cuda'

        self.supervise_sp_offset = getattr(param_loss, "supervise_sp_offset", True)

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.occupany_L1loss = nn.L1Loss()

        self.instance_size_L1loss = nn.L1Loss()

        self.superpoint_semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
    
    def forward(self, loss_inp, epoch):
        loss_out = {}
        
        ###################################################################################################
        # ---------------------- point-level ----------------------
        semantic_labels, instance_labels = loss_inp['point_labels']
        point_valid = (instance_labels != self.ignore_label)

        # ---------------------- point semantic --------------------
        semantic_scores = loss_inp["semantic_scores"]
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)
        if self.semantic_dice:
            filter_ids = (semantic_labels != self.ignore_label)
            semantic_scores = semantic_scores[filter_ids]
            semantic_scores = F.softmax(semantic_scores, dim=-1)
            semantic_labels = semantic_labels[filter_ids]
            one_hot_labels = F.one_hot(semantic_labels, num_classes=self.semantic_class_num)
            semantic_loss += dice_loss_multi_classes(semantic_scores, one_hot_labels).mean()
        loss_out["semantic_loss"] = (semantic_loss, semantic_scores.sum())

        ###################################################################################################
        if epoch > self.joint_training_epoch: 
            # ---------------------- superpoint-level -------------------------
            superpoint_semantic_labels, superpoint_instance_labels = loss_inp['superpoint_labels']
            sp_valid = ((superpoint_instance_labels != self.ignore_label) & (superpoint_semantic_labels != self.ignore_label))

            # ---------------------- superpoint semantic --------------------
            sp_semantic_scores = loss_inp['sp_semantic']
            superpoint_semantic_loss = self.superpoint_semantic_criterion(sp_semantic_scores, superpoint_semantic_labels)
            loss_out["superpoint_semantic_loss"] = (superpoint_semantic_loss, sp_semantic_scores.sum())

            # ---------------------- superpoint offset vector --------------------
            if self.supervise_sp_offset:
                pred_sp_offset_vectors, superpoint_offset_vector = loss_inp['sp_offset_vector']
                
                pt_diff = pred_sp_offset_vectors - superpoint_offset_vector   # [N, 3]
                pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # [N]

                offset_norm_loss = torch.sum(pt_dist * sp_valid) / (sp_valid.sum() + 1e-6)

                gt_offsets_norm = torch.norm(superpoint_offset_vector, p=2, dim=1)   # [N], float
                gt_offsets_ = superpoint_offset_vector / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
                pt_offsets_norm = torch.norm(pred_sp_offset_vectors, p=2, dim=1)
                pt_offsets_ = pred_sp_offset_vectors / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
                direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # [N]
                offset_dir_loss = torch.sum(direction_diff * sp_valid) / (sp_valid.sum() + 1e-6)

                loss_out["offset_norm_loss"] = (offset_norm_loss, sp_valid.sum())
                loss_out["offset_dir_loss"] = (offset_dir_loss, sp_valid.sum())

            # ---------------------- superpoint discriminative loss ------------------
            sp_discriminative_feats, sp_batch_offsets = loss_inp['sp_discriminative_features']
            sp_d_loss, sp_l_var, sp_l_dist, sp_l_reg = [], [], [], []
            for i in range(1, len(sp_batch_offsets)):
                begin, end = sp_batch_offsets[i-1], sp_batch_offsets[i]
                _discriminative_feats, _instance_labels, _sp_valid = sp_discriminative_feats[begin:end], superpoint_instance_labels[begin:end], sp_valid[begin:end]
                _discriminative_feats, _instance_labels = _discriminative_feats[_sp_valid], _instance_labels[_sp_valid]
                assert len(_discriminative_feats) == len(_instance_labels)
                
                d_loss, l_var, l_dist, l_reg = self.discriminative_loss(_discriminative_feats, _instance_labels) 
                
                sp_d_loss.append(d_loss.view(-1))
                sp_l_var.append(l_var.view(-1))
                sp_l_dist.append(l_dist.view(-1))
                sp_l_reg.append(l_reg.view(-1))
            sp_d_loss, sp_l_var, sp_l_dist, sp_l_reg = torch.cat(sp_d_loss), torch.cat(sp_l_var), torch.cat(sp_l_dist), torch.cat(sp_l_reg)
            sp_d_loss, sp_l_var, sp_l_dist, sp_l_reg = torch.mean(sp_d_loss), torch.mean(sp_l_var), torch.mean(sp_l_dist), torch.mean(sp_l_reg)
            loss_out['superpoint_discriminative_loss'] = (sp_d_loss, sp_discriminative_feats.shape[0])

            # ------------------------ instance size loss -------------------------
            if self.supervise_instance_size: 

                pred_sp_occupancy, superpoint_instance_voxel_num = loss_inp['sp_occupancy']
                pred_sp_occupancy, superpoint_instance_voxel_num = pred_sp_occupancy[sp_valid], superpoint_instance_voxel_num[sp_valid]
                occupancy_loss = self.occupany_L1loss(pred_sp_occupancy, superpoint_instance_voxel_num)
                loss_out['occupancy_loss'] = (occupancy_loss, sp_valid.sum())

                pred_sp_instance_size, superpoint_instance_size = loss_inp['sp_instance_size']
                pred_sp_instance_size, superpoint_instance_size = pred_sp_instance_size[sp_valid], superpoint_instance_size[sp_valid]
                instance_size_loss = self.instance_size_L1loss(pred_sp_instance_size, superpoint_instance_size)
                loss_out['instance_size_loss'] = (instance_size_loss, sp_valid.sum())

        # ----------------------- loss sum --------------------------
        
        loss = 0.0
        loss += 1.0 * semantic_loss # 3D U-Net
        self.logger.info('point semantic loss: {:.4f}'.format(semantic_loss))

        if epoch > self.joint_training_epoch: 
            loss += 1.0 * superpoint_semantic_loss
            self.logger.info('sp semantic loss: {:.4f}'.format(superpoint_semantic_loss))

            if self.supervise_sp_offset:
                loss += (1.0 * offset_norm_loss + 1.0 * offset_dir_loss)
                self.logger.info('sp offset norm loss: {:.4f}'.format(offset_norm_loss))
                self.logger.info('sp offset dir loss: {:.4f}'.format(offset_dir_loss))

            loss += 1.0 * sp_d_loss
            self.logger.info('sp discriminative loss: {:.4f}'.format(sp_d_loss))

            if self.supervise_instance_size:
                loss += 1.0 * occupancy_loss
                self.logger.info('sp occupancy loss: {:.4f}'.format(occupancy_loss))
                loss += 1.0 * instance_size_loss
                self.logger.info('sp instance size loss: {:.4f}'.format(instance_size_loss))

        return loss, loss_out



    def discriminative_loss(self, prediction, correct_label):

        reshaped_pred = torch.reshape(prediction, [-1, self.discriminative_feature_dim])

        unique_labels, unique_id, counts = torch.unique(correct_label, sorted=False, return_inverse=True, return_counts=True)


        counts = counts.float()
        num_instances = torch.tensor(unique_labels.size()[0])
        # if num_instances <= 1:
        #     return torch.tensor(0.).float().to(self.device), torch.tensor(0.).float().to(self.device), torch.tensor(0.).float().to(self.device), torch.tensor(0.).float().to(self.device)

        segmented_sum = torch.zeros(num_instances, self.discriminative_feature_dim).to(self.device) 
        segmented_sum = segmented_sum.index_add_(0, unique_id, reshaped_pred) 


        mu = torch.div(segmented_sum, torch.reshape(counts, (-1, 1))) 
        mu_expand = mu[unique_id]



        tmp_distance = reshaped_pred - mu_expand
        dist = torch.norm(tmp_distance, p=2, dim=1)
 
        dist = dist - self.delta_v
        dist = torch.clamp(dist, min=0.) 
        dist = torch.square(dist)


        l_var = torch.zeros(num_instances).to(self.device) 
        l_var = l_var.index_add_(0, unique_id, dist)
        l_var = torch.div(l_var, counts) 
        l_var = torch.sum(l_var)
        l_var = torch.div(l_var, num_instances) 
        # -------- discriminative_loss l_var ----------

        # -------- discriminative_loss l_dist -----------

        
        if num_instances <= 1: 

            l_dist = torch.tensor(0.).float().to(self.device) 
        else:
            mu_diff_norm1 = torch.cdist(mu, mu, p=1)
            
            mu_diff_norm1 = 2. * self.delta_d - mu_diff_norm1
            
            mu_diff_norm1 = mu_diff_norm1 - torch.diagflat(torch.diag(mu_diff_norm1, 0))
            
            mu_diff_norm1 = torch.clamp(mu_diff_norm1, min=0.)
            
            mu_diff = torch.square(mu_diff_norm1)
            
            l_dist = torch.sum(mu_diff)
            l_dist = torch.div(l_dist, num_instances * (num_instances - 1))
        

        l_reg = torch.sum(torch.norm(mu, p=2, dim=1))
 

        l_var = self.param_var * l_var
        l_dist = self.param_dist * l_dist
        l_reg = self.param_reg * l_reg

        loss = l_var + l_dist + l_reg

        if torch.isnan(loss).any():
            print('label :', torch.isnan(correct_label).any())
            print(correct_label)
            print('input :', torch.isnan(reshaped_pred).any())
            print(reshaped_pred)
            return
        
        return loss, l_var, l_dist, l_reg


def dice_loss_multi_classes(input: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float = 1e-5,
                            weight: Optional[float]=None) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(axis_order)
    target = target.permute(axis_order)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                       (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1. - per_channel_dice
    
    return loss