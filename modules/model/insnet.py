# Copyright (c) Gorilla-Lab. All rights reserved.
import functools
from typing import Dict
import ast

import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_max

import pointgroup_ops

from sparse_unet3d import UBlock, ResidualBlock
from fc import MultiFC
from gcn import GCN

from func_helper import *
import utils

__all__ = ['Network']

class Network(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.input_channel = param.input_channel                                # 3
        self.use_coords = param.use_coords                                      # True
        self.blocks = param.blocks                                              # 5
        self.block_reps = param.block_reps                                      # 2
        self.media = param.media                                                # 32
        self.classes = param.classes                                            # 20
        self.score_scale = param.score_scale                                    # 50
        self.score_fullscale = param.score_fullscale                            # 14
        self.mode = param.score_mode                                            # 4 == mean
        self.detach = param.detach                                              # True
        self.affinity_weight = list(map(float, ast.literal_eval(param.affinity_weight)))    # List[float]=[1.0, 1.0]
        self.with_refine = param.with_refine                                    # False
        self.fusion_epochs = param.fusion_epochs                                # 128
        self.score_epochs = param.score_epochs                                  # 160
        self.fix_module = ast.literal_eval(param.fix_module)                    # List[str]=[]

        # -------------------------- backbone -------------------------
        if self.use_coords:
            self.input_channel += 3

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channel, self.media, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        block_list = [self.media * (i + 1) for i in range(self.blocks)]
        self.unet = UBlock(block_list, norm_fn, self.block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(self.media),
            nn.ReLU(inplace=True)
        )

        # --------------------- semantic segmentation ---------------------
        self.linear = nn.Sequential(
            nn.Linear(self.media, self.media, bias=True),
            norm_fn(self.media),
            nn.ReLU(inplace=True),
            nn.Linear(self.media, self.classes)
        )

        # ----------------------------- offset ----------------------------
        self.offset = nn.Sequential(
            nn.Linear(self.media, self.media, bias=True),
            norm_fn(self.media),
            nn.ReLU(inplace=True)
        )
        self.offset_linear = nn.Linear(self.media, 3, bias=True)

        # -------------------------- fusion layer ------------------------
        fusion_channel = 2 * (self.media + self.classes + 3 * 2)
        refine_channel = self.media + self.classes + 3 * 2

        # --------------- superpoint, fusion and refine branch -----------
        self.superpoint_linear = MultiFC(
            nodes = [self.media, self.media, self.media],
            drop_last = False
        )
        self.fusion_linear = MultiFC(
            nodes = [fusion_channel, 256, 512, 512, 1]
        )
        self.refine_gcn = GCN([refine_channel, 128, 128, 1])

        # -------------------------- score branch ------------------------
        self.score_unet = UBlock([self.media, 2 * self.media], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(self.media),
            nn.ReLU(inplace=True)
        )
        self.score_linear = nn.Linear(self.media, 1)

        self.apply(self.set_bn_init)

        # -------------------------- fix parameter ------------------------
        for module in self.fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def freeze_bn(module):
        for name, child in module._modules.items():
            if child is not None:
                Network.freeze_bn(child)
            if isinstance(child, nn.BatchNorm1d):
                if hasattr(child, "weight"):
                    child.weight.requires_grad_(False)
                if hasattr(child, "bias"):
                    child.bias.requires_grad_(False)

    @staticmethod
    def set_bn_init(m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            try:
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
            except:
                pass


    def clusters_voxelization(self,
                              clusters_idx: torch.Tensor,
                              feats: torch.Tensor,
                              coords: torch.Tensor,
                              fullscale: int,
                              scale: int,
                              mode: int):
        r"""voxelize clusters(instance proposals)

        Args:
            clusters_idx (torch.Tensor, [sumNPoint, 2]): , int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
            feats (torch.Tensor, [N, C]): features of points
            coords (torch.Tensor, [N, 3]): coordinates of points
            fullscale (int): the range of voxelization(for dynamic)
            scale (int): the clamp of voxelization
            mode (int): voxelization pooling mode
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]
        
        # https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/mean.html
        clusters_coords_mean = scatter_mean(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0) # [nCluster, 3], float

        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # [sum_points, 3], float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = scatter_min(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0)[0] # [nCluster, 3], float
        clusters_coords_max = scatter_max(clusters_coords, clusters_idx[:, 0].cuda().long(), dim=0)[0] # [nCluster, 3], float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # [nCluster], float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # [nCluster, 3], float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # [sum_points, 1 + 3]

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * [1 + 3] long
        # input_map: sum_points int
        # output_map: M * [maxActive + 1] int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # [M, C], float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


    def hierarchical_fusion_step(self,
                                 superpoint: torch.Tensor,
                                 batch_idxs: torch.Tensor,
                                 coords: torch.Tensor,
                                 features: torch.Tensor,
                                 semantic_scores: torch.Tensor,
                                 pt_offsets: torch.Tensor,
                                 instance_labels: Optional[torch.Tensor]=None,
                                 scene: Optional[str]=None,
                                 prepare_flag: bool=False,
                                 ret: Optional[dict]=None,
                                 mode: str="train") -> None:
        r"""
        the core step, building hierarchical clustering tree and bfs traverse tree
        to get all fusion pair and judge whether fusion or not

        Args:
            superpoint (torch.Tensor, [N]): superpoint id of points
            batch_idxs (torch.Tensor, [N]): batch idx of points
            coords (torch.Tensor, [N, 3]): coordinates of points
            features (torch.Tensor, [N, C]): features of points
            semantic_scores (torch.Tensor, [N, classes]): semantic scores of points
            pt_offsets (torch.Tensor, [N, 3]): predict offsets of points
            instance_labels (Optional[torch.Tensor], [N]): instance label id of points. Defaults to None.
            scene (Optional[str], optional): scene name. Defaults to None.
            prepare_flag (bool, optional): prepare or not. Defaults to False.
            ret (Optional[dict], optional): dict to save results. Defaults to None.
            mode (str, optional): train or test mode. Defaults to "train".
        """
        timer = utils.Timer()
        # count the superpoint num
        shifted = coords + pt_offsets               # [N, 3]
        
        semantic_scores = F.softmax(semantic_scores, dim=-1)                        # [N, nClasses]
        semantic_preds = semantic_scores.max(1)[1]                                  # [N], long, index
        # print(f'semantic_preds: {semantic_preds.shape} {semantic_preds.dtype}')
        semantic_preds = voting_semantic_segmentation(semantic_preds, superpoint)   # [N], float32
        # print(f'semantic_preds: {semantic_preds.shape} {semantic_preds.dtype}')
        
        # get point-wise affinity
        semantic_weight, shifted_weight = self.affinity_weight  # 1.0, 1.0
        affinity = torch.cat([semantic_scores * semantic_weight, shifted * shifted_weight], dim=1)
        # affinity: [N, nClasses]*1.0; [N, 3] => [N, nClasses+3] = [N, C']
        append_feaures = torch.cat([semantic_scores, shifted, coords], dim=1)
        # append_feaures:  [N, nClasses]; [N, 3]; [N, 3] => [N, nClasses+3+3]

        if self.detach:
            append_feaures = append_feaures.detach()
        
        # filter out according to semantic prediction labels
        filter_ids = torch.nonzero(semantic_preds > 1).view(-1)     # 0, 1 == floor and wall, [N']
        
        print(f'superpoint: {superpoint.shape}')
        """
        print(f'semantic_preds: {semantic_preds.shape}')
        print(f'superpoint: {superpoint.shape}')
        print(f'min(superpoint) {torch.min(superpoint)} max(superpoint) {torch.max(superpoint)}')
        tmp1, idx1 = torch.unique(superpoint, return_inverse=True)
        print(f'tmp1: {tmp1.shape} idx1: {idx1.shape}')
        print(f'min(tmp1) {torch.min(tmp1)} min(idx1) {torch.min(idx1)}')
        print(f'max(tmp1) {torch.max(tmp1)} max(idx1) {torch.max(idx1)}')
        exit()
        superpoint = torch.unique(superpoint[filter_ids], return_inverse=True)[1]   # [N']
        coords = coords[filter_ids]                 # [N', 3]
        features = features[filter_ids]             # [N', C]
        affinity = affinity[filter_ids]             # [N', C']
        append_feaures = append_feaures[filter_ids] # [N', C']
        semantic_preds = semantic_preds[filter_ids] # [N']
        batch_idxs = batch_idxs[filter_ids]         # [N']
        shifted = shifted[filter_ids]               # [N', 3]
        """
        # get the superpoint-wise input
        # https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/mean.html
        # scatter_mean(src, index, dim)
        # get mean results according to the index
        superpoint_affinity = scatter_mean(affinity, superpoint, dim=0)             # [Nsp, C], Nsp: the number of superpoints
        superpoint_batch_idxs = scatter_mean(batch_idxs, superpoint, dim=0).int()   # [Nsp],  torch.int32
        superpoint_centers = scatter_mean(coords, superpoint, dim=0)                # [Nsp, 3]
        superpoint_features = scatter_mean(features, superpoint, dim=0)             # [Nsp, C]
        superpoint_features = self.superpoint_linear(superpoint_features)           # [Nsp, C]: three-layer FC
        superpoint_append_feaures = scatter_mean(append_feaures, superpoint, dim=0) # [Nsp, C']
        superpoint_count = torch.bincount(superpoint)                               # [Nsp]
        # input:                    [4, 3, 6, 3, 4]
        # torch.bincount(input):    [0, 0, 0, 2, 2, 0, 1]
        # print(f'superpoint_count: {superpoint_count.shape}')
        # print(f'superpoint_count min: {torch.min(superpoint_count)}')
        # print(f'superpoint_count max: {torch.max(superpoint_count)}')
        # tmp = torch.nonzero(superpoint_count > 0).view(-1)
        # print(f'tmp: {tmp.shape}')


        if mode == "train":
            # instance_labels = instance_labels[filter_ids]     # [N']
            # instance_labels                                   # [N]
            num_inst = int(instance_labels.max() + 1)           # number of instance
            # print(f'instance_labels: {instance_labels.shape}')
            # print(f'instance_labels: min {torch.min(instance_labels)} max {torch.max(instance_labels)}')
            print(f'num_inst: {num_inst}')
            _, superpoint_soft_inst_label = align_superpoint_label(instance_labels, superpoint, num_inst) # [num_superpoint], [num_superpoint, num_inst + 1]
            # superpoint_soft_inst_label: [Nsp, num_inst + 1], 1 for ignored label -100
            # print(f'superpoint_soft_inst_label: {superpoint_soft_inst_label.shape}')
        else:
            superpoint_soft_inst_label = superpoint_features

        superpoint_features = torch.cat([superpoint_features, superpoint_append_feaures], dim=1)
        # superpoint_features: [Nsp, C + C']

        print(f'superpoint_batch_idxs: {superpoint_batch_idxs.shape}')
        print(f'superpoint_batch_idxs: {superpoint_batch_idxs.dtype}')
        bidx = torch.unique(superpoint_batch_idxs)
        print(f'bidx: {bidx.shape}')
        print(f'bidx: {bidx}')

        exit()
        # bottom-up build the hierarchical tree
        hierarchical_tree_list, tree_list, fusion_features_list, fusion_labels_list, nodes_list = build_hierarchical_tree(
            superpoint_affinity,            # [Nsp, nClass]     => [semantic_score; coord+offset]
            superpoint_features,            # [Nsp, C]          => superpoint features
            superpoint_centers,             # [Nsp, 3]          => superpoint center
            superpoint_count,               # [Nsp]             => the number of points in each superpoint
            superpoint_batch_idxs,          # [Nsp]             => record the batch_idx of each superpoint
            superpoint_soft_inst_label      # [Nsp, num_inst+1] or [Nsp, C]
        )

        # construct batch idx
        batch_idx_list = []
        for batch_idx in torch.unique(batch_idxs):
            batch_idx_list.extend([batch_idx] * len(fusion_features_list[batch_idx]))
        batch_idxs = torch.Tensor(batch_idx_list).to(fusion_features_list[0].device) # [num_fusion]
        
        # predict fusion scores(for split)
        scores_features = torch.cat(fusion_features_list) # [num_fusion, C]
        fusion_scores = self.fusion_linear(scores_features).squeeze() # [num_fusion]
        batch_idxs = torch.Tensor(batch_idx_list).to(fusion_scores.device) # [num_fusion]
        fusion_labels = torch.cat(fusion_labels_list)
        ret["fusion"] = (fusion_scores, fusion_labels)
        
        # top-down traversal and split
        if self.with_refine or prepare_flag:
            proposals_idx_bias = 0
            proposals_idx_list = []
            refine_scores_list = []
            refine_labels_list = []
            # loop batch
            for batch_idx in torch.unique(batch_idxs):
                batch_idx = int(batch_idx)
                tree = tree_list[batch_idx]
                nodes = nodes_list[batch_idx]
                ids = (batch_idxs == batch_idx)
                batch_fusion_scores = torch.sigmoid(fusion_scores[ids])
                threshold = 0.5
                batch_fusion_labels = (batch_fusion_scores[::2] > threshold) & (batch_fusion_scores[1::2] > threshold)
                batch_fusion_labels = batch_fusion_labels.cpu().numpy().tolist()
                node_ids = nodes.int().cpu().numpy().tolist()
                # traversal with splitting
                batch_cluster_list, batch_node_ids, refine_labels = \
                    traversal_cluster(tree, node_ids, batch_fusion_labels)

                empty_flag = refine_labels is None
                ret["empty_flag"] = empty_flag

                if self.with_refine and not empty_flag:
                    batch_adjancy_matrix = build_superpoint_clique(tree, batch_node_ids)
                    batch_adjancy_matrix = batch_adjancy_matrix.cuda()
                    num_leaves = len(tree.leaves(tree.root))
                    input_features = [tree.get_node(i).data.feature for i in range(num_leaves)]
                    for node_id in batch_node_ids:
                        input_features.append(tree.get_node(node_id).data.feature)
                    batch_graph_input_features = torch.stack(input_features)
                    # get refine scores
                    batch_refine_scores = self.refine_gcn(batch_graph_input_features, batch_adjancy_matrix).squeeze()
                    # batch_cluster_ids = gorilla.concat_list(batch_cluster_list)   # le
                    batch_cluster_ids = utils.concat_list(batch_cluster_list)
                    refine_scores = batch_refine_scores[batch_cluster_ids]
                    refine_scores_list.append(refine_scores)
                    refine_labels_list.append(refine_labels)
                    if prepare_flag:
                        refine_cluster_list = []
                        start = 0
                        end = 0
                        refine_scores = torch.sigmoid(refine_scores).detach().cpu().numpy()
                        # refine_scores = refine_labels.cpu().numpy()
                        refinement = (refine_scores > 0.5)
                        for cluster in batch_cluster_list:
                            end += len(cluster)
                            cluster_refinement = refinement[start: end]
                            start += len(cluster)
                            refine_cluster = np.array(cluster)[cluster_refinement].tolist()
                            if len(refine_cluster) == 1:
                                continue
                            refine_cluster_list.append(refine_cluster)
                        batch_cluster_list = refine_cluster_list

                if not empty_flag:
                    # get proposals
                    # convert superpoint id cluster into mask proposals
                    proposals_idx = get_proposals_idx(superpoint, batch_cluster_list)
                    proposals_idx[:, 1] = filter_ids.cpu()[proposals_idx[:, 1]]
                    proposals_idx[:, 0] += proposals_idx_bias
                    proposals_idx_bias = (proposals_idx[:, 0].max() + 1)
                    proposals_idx_list.append(proposals_idx)

            if self.with_refine and not empty_flag:
                refine_scores = torch.cat(refine_scores_list)
                refine_labels = torch.cat(refine_labels_list)
                ret["refine"] = (refine_scores, refine_labels)

            if prepare_flag and not empty_flag:
                proposals_idx = torch.cat(proposals_idx_list) # [num_proposals, 2]
                proposals_offset = get_batch_offsets(proposals_idx[:, 0], proposals_idx[:, 0].max() + 1)
                ret["proposals"] = proposals_idx, proposals_offset

    def forward(self,
                input: spconv.SparseConvTensor,
                input_map: torch.Tensor,
                coords: torch.Tensor,
                epoch: int,
                extra_data: Optional[Dict]=None,
                mode: str="train",
                semantic_only: bool=False) -> Dict:
        r"""SSTNet forward

        Args:
            input (spconv.SparseConvTensor): input sparse tensor
            input_map (torch.Tensor, [N]): points to voxels indices
            coords (torch.Tensor, [N, 3]): coordinates of points
            epoch (int): epoch number
            extra_data (Optional[Dict], optional): extra data dict. Defaults to None.
            mode (str, optional): forward mode. Defaults to "train".
            semantic_only (bool, optional): predict semantic results or not. Defaults to False.

        Returns:
            Dict: prediction results
        """
        timer = utils.Timer()
        ret = {}
        for module in self.fix_module:
            getattr(self, module).eval()

        # ---------------------- U-Net backbone head ---------------------
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]        # [N, C] : [M, C] -> [N, C]
        # print(f'output_feats: {output_feats.shape}')

        

        # ----------------- HEAD: semantic segmentation -------------------
        semantic_scores = self.linear(output_feats)             # [N, nClass], float
        ret["semantic_scores"] = semantic_scores

        # -------------------------- HEAD: offset -------------------------
        pt_offsets_feats = self.offset(output_feats)            # [N, C]
        pt_offsets = self.offset_linear(pt_offsets_feats)       # [N, 3], float32
        ret["pt_offsets"] = pt_offsets

        # --------------------- HEAD: superpoint ------------------------
        # deal with superpoint

        batch_idxs = extra_data["batch_idxs"]                   # [N]
        # print(f'batch_idxs: {batch_idxs.shape}')
        scene_list = extra_data["scene_list"]                   # (batch_size)
        # print(f'scene_list: {scene_list}')
        superpoint = extra_data["superpoint"]                   # [N]
        # print(f'superpoint: {superpoint.shape}')
        
        instance_labels = None
        if mode == "train":
            instance_labels = extra_data["instance_labels"]     # [N]
            # print(f'instance_labels: {instance_labels.shape}')

        prepare_flag = epoch > self.score_epochs                # score_epochs: 160
        fusion_flag = epoch > self.fusion_epochs                # fusion_epochs: 128

        ret["empty_flag"] = False
        # hierarchical clustering and get proposal

        # if epoch > fusion_epochs (default: 128) then perform 
        if fusion_flag and not semantic_only:
            self.hierarchical_fusion_step(superpoint,
                                          batch_idxs,
                                          coords,
                                          output_feats,
                                          semantic_scores,
                                          pt_offsets,
                                          instance_labels,
                                          scene=scene_list[0],
                                          prepare_flag=prepare_flag,
                                          ret=ret,
                                          mode=mode)

        # ----------------------- HEAD: score ----------------------------
        # if epoch > score_epochs (default: 160) then predict score

        if prepare_flag and not semantic_only:
            proposals_idx, proposals_offset = ret["proposals"]

            #### proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, output_feats, coords, self.score_fullscale, self.score_scale, self.mode)

            #### score
            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()]  # [sum_points, C]
            score_feats = scatter_max(score_feats, proposals_idx[:, 0].cuda().long(), dim=0)[0] # [num_prop, C]
            proposal_scores = torch.sigmoid(self.score_linear(score_feats).squeeze()) # [num_prop]

            ret["proposal_scores"] = proposal_scores

        return ret


def get_batch_offsets(batch_idxs: torch.Tensor, bs: int) -> torch.Tensor:
    r"""get the offsets of batch ids

    Args:
        batch_idxs (torch.Tensor, [N]): batch ids
        bs (int): number of batch size

    Returns:
        torch.Tensor, [bs + 1]: batch offsets
    """
    batch_idxs_np = batch_idxs.cpu().numpy()
    batch_offsets = np.append(np.searchsorted(batch_idxs_np, range(bs)), len(batch_idxs_np))
    batch_offsets = torch.Tensor(batch_offsets).int().to(batch_idxs.device)
    return batch_offsets

