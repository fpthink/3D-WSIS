
import functools
from typing import Dict
import ast


import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_max, scatter

import pointgroup_ops

from sparse_unet3d import UBlock, ResidualBlock
from fc import MultiFC
from gcn import GCN
import graphnet 

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
        self.fix_module = ast.literal_eval(param.fix_module)                    # List[str]=[]

        #### backbone
        
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


        # #### semantic segmentation
        self.linear = nn.Sequential(
            nn.Linear(self.media, self.media, bias=True),
            norm_fn(self.media),
            nn.ReLU(inplace=True),
            nn.Linear(self.media, self.classes)
        )


        edge_feats_dim = 13
        self.ecc = graphnet.GraphNetwork('gru_7_0,f_64,b,r', nfeat=self.media, fnet_widths=[edge_feats_dim] + [32,128,64],
                                            fnet_orthoinit=True, fnet_llbias=True, fnet_bnidx=2, use_pyg=True, cuda=True) 

        sp_feat_dim = 64

        #### superpoint semantic segmentation
        self.sp_sem_seg = nn.Sequential(
            nn.Linear(sp_feat_dim, sp_feat_dim, bias=True),
            norm_fn(sp_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sp_feat_dim, self.classes)
        )


        #### superpoint offset vector 
        self.sp_offset_vector_head = nn.Sequential(
            nn.Linear(sp_feat_dim, sp_feat_dim, bias=True),
            norm_fn(sp_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sp_feat_dim, 3)
        )


        #### superpoint instance voxel num # 2021.12.21
        self.sp_occupancy_head = nn.Sequential(
            nn.Linear(sp_feat_dim, sp_feat_dim, bias=True),
            norm_fn(sp_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sp_feat_dim, 1)
        )


        ### superpoint instance radius
        self.sp_ins_size_head = nn.Sequential(
            nn.Linear(sp_feat_dim, sp_feat_dim, bias=True),
            norm_fn(sp_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sp_feat_dim, 1)
        )
        
        #### self-attention for affinity matrix
        d_model = 64
        self.fc_position = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )  

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)


        self.feature_term = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            norm_fn(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 7)  
        )



        #### fix parameter
        # print('fix U-net param ...')
        # self.fix_module = ['input_conv', 'unet', 'output_layer']
        for module in self.fix_module:
            module = getattr(self, module)
            print(module)
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
    
   


    def forward(self,
                input: spconv.SparseConvTensor,
                input_map: torch.Tensor,
                extra_data,
                ) -> Dict:


        ret = {}
        for module in self.fix_module:
            getattr(self, module).eval()

        output = self.input_conv(input)
        output = self.unet(output)

        output = self.output_layer(output)
        output_feats = output.features[input_map.long()] # [N, m] #  voxel-level to point-level

        #### point-level semantic segmentation
        semantic_scores = self.linear(output_feats)   # [N, nClass], float
        ret["semantic_scores"] = semantic_scores


        superpoint = extra_data["superpoint"] # [N]
        superpoint = superpoint.long()  
        embeddings = scatter(output_feats, superpoint, dim=0, reduce='mean')
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html

        GIs = extra_data['GIs']
        self.ecc.set_info(GIs, cuda=True)
        ecc_outputs = self.ecc(embeddings)

        sp_semantic_scores = self.sp_sem_seg(ecc_outputs)
        ret['sp_semantic_scores'] = sp_semantic_scores
        
        pred_sp_offset_vectors = self.sp_offset_vector_head(ecc_outputs)
        ret['pred_sp_offset_vectors'] = pred_sp_offset_vectors

        pred_sp_occupancy = self.sp_occupancy_head(ecc_outputs)
        ret['pred_sp_occupancy'] = pred_sp_occupancy.squeeze(-1) # (N, 1) reshape => (N, )

        pred_sp_ins_size = self.sp_ins_size_head(ecc_outputs) 
        ret['pred_sp_ins_size'] = pred_sp_ins_size.squeeze(-1)


        ##################### cal affinity matrix by edge ###################
        superpoint_center_xyz = extra_data['superpoint_cenetr_xyz'] # N x 3
        q, k, v = self.w_qs(ecc_outputs), self.w_ks(ecc_outputs), self.w_vs(ecc_outputs)


        edge_u_list = extra_data["edge_u_list"]
        edge_v_list = extra_data["edge_v_list"]

        pos_enc = self.fc_position(superpoint_center_xyz[edge_u_list] - superpoint_center_xyz[edge_v_list]).reshape(-1) # E x 1

        affinity = (q[edge_u_list] * k[edge_v_list]).sum(dim=1)

        affinity = affinity / np.sqrt(k.size(-1))

        affinity = affinity * pos_enc

        # --------- avoid numeric overflow & underflow ----------
        _max = scatter(affinity, edge_u_list, dim=0, reduce='max')
        _max = _max[edge_u_list]
        affinity = affinity - _max
        # -------------------------------

        exp_affinity = torch.exp(affinity)

        total_exp = scatter(exp_affinity, edge_u_list, dim=0, reduce='sum')
        
        total_exp = total_exp[edge_u_list]
        
        affinity = exp_affinity / total_exp 

        ret['edge_affinity'] = affinity

        affinity = affinity.reshape(-1, 1)

        res = affinity * v[edge_v_list] 

        res = scatter(res, edge_u_list, dim=0, reduce='sum')

        sp_feat = torch.zeros(ecc_outputs.shape).to(ecc_outputs)
        sp_feat = sp_feat + ecc_outputs 

        sp_feat[:res.shape[0]] += res 

        sp_discriminative_features = self.feature_term(sp_feat)

        ret['sp_discriminative_feats'] = sp_discriminative_features

        return ret


