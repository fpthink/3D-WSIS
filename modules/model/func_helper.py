# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
from torch_scatter import scatter_add
from treelib import Tree

import htree
from cluster.hierarchy import linkage


class Node:
    def __init__(self,
                 feature: torch.Tensor,
                 center: torch.Tensor,
                 soft_label: Optional[torch.Tensor]=None,
                 num: Optional[int]=1) -> None:
        super().__init__()
        self.feature = feature
        self.center = center
        self.soft_label = soft_label
        self.num = num


def build_hierarchical_tree(affinity: torch.Tensor,
                            features: torch.Tensor,
                            centers: torch.Tensor,
                            affinity_count: torch.Tensor,
                            batch_idxs: torch.Tensor,
                            soft_label: Optional[torch.Tensor]=None):
    r"""
    build the hierarchical tree

    Args:
        affinity (torch.Tensor, [num_leaves, C]): affinity of nodes
        features (torch.Tensor, [num_leaves, C']): features of nodes
        centers (torch.Tensor, [num_leaves, 3]): centers of nodes
        affinity_count (torch.Tensor, [num_leaves]): point count of nodes
        batch_idxs (torch.Tensor, [num_leaves]): batch idxs of nodes
        soft_label (Optional[torch.Tensor], [num_leaves, num_label + 1]): soft label of nodes. Default to None

    Returns:
        list of tree and tree_connection
    """
    tree_list = []
    hierarchical_tree_list = []
    scores_features_list = []
    labels_list = []
    nodes_list = []
    # build hierarchical tree for each batch
    for batch_idx in torch.unique(batch_idxs):
        ids = (batch_idxs == batch_idx)
        num_batch = ids.sum()
        batch_centers = centers[ids]        # [num_batch, 3]
        batch_affinity = affinity[ids]      # [num_batch, C]
        batch_features = features[ids]      # [num_batch, C']
        batch_soft_label = soft_label[ids]  # [num_batch, num_inst+1] or [num_batch, C]

        # build tree by affinity
        batch_affinity_count = affinity_count[ids]              # [num_batch]
        affinity_np = batch_affinity.detach().cpu().numpy()     # (num_batch, C)
        affinity_np = np.concatenate([affinity_np, batch_affinity_count[:, None].cpu().numpy()], axis=1)
        # (num_leaves, C+1)
        tree_connection = linkage(affinity_np, method="average", with_observation=True)
        tree_connection = tree_connection[:, :2].astype(np.int)

        # add leaf nodes
        node_list = [Node(batch_features[i],
                          batch_centers[i],
                          batch_soft_label[i],
                          batch_affinity_count[i]) for i in range(num_batch)]
        
        num_nodes = tree_connection.max() + 1

        connection = tree_connection.tolist()
        hierarchical_tree = htree.Tree(connection)
        hierarchical_tree_list.append(hierarchical_tree)

        # get the fusion process and cuda
        left_leaves, left_ids, right_leaves, right_ids, fusion_leaves, fusion_ids = hierarchical_tree.fusion_record()
        left_leaves = torch.Tensor(left_leaves).long().to(affinity.device)
        left_ids = torch.Tensor(left_ids).long().to(affinity.device)
        right_leaves = torch.Tensor(right_leaves).long().to(affinity.device)
        right_ids = torch.Tensor(right_ids).long().to(affinity.device)
        fusion_leaves = torch.Tensor(fusion_leaves).long().to(affinity.device)
        fusion_ids = torch.Tensor(fusion_ids).long().to(affinity.device)

        # record the fusion nodes
        fusion_affinity_count = scatter_add(batch_affinity_count[fusion_leaves], fusion_ids, dim=0) # [num_fusion]
        node_features = get_fusion_property(batch_features, batch_affinity_count, fusion_leaves, fusion_ids, fusion_affinity_count)
        node_centers = get_fusion_property(batch_centers, batch_affinity_count, fusion_leaves, fusion_ids, fusion_affinity_count)
        node_soft_label = get_fusion_property(batch_soft_label, batch_affinity_count, fusion_leaves, fusion_ids, fusion_affinity_count)

        # addd intermidiate tree node
        for i in range(len(connection)):
            node_list.append(Node(
                            node_features[i],
                            node_centers[i],
                            node_soft_label[i],
                            fusion_affinity_count[i]))

        # get the left and right children's property for each node
        left_affinity_count = scatter_add(batch_affinity_count[left_leaves], left_ids, dim=0) # [num_fusion]
        left_features = get_fusion_property(batch_features, batch_affinity_count, left_leaves, left_ids, left_affinity_count)
        left_soft_label = get_fusion_property(batch_soft_label, batch_affinity_count, left_leaves, left_ids, left_affinity_count)
        right_affinity_count = scatter_add(batch_affinity_count[right_leaves], right_ids, dim=0) # [num_fusion]
        right_features = get_fusion_property(batch_features, batch_affinity_count, right_leaves, right_ids, right_affinity_count)
        right_soft_label = get_fusion_property(batch_soft_label, batch_affinity_count, right_leaves, right_ids, right_affinity_count)

        features_list = torch.cat([torch.cat([left_features, right_features], dim=1)[:, None, :],
                                   torch.cat([right_features, left_features], dim=1)[:, None, :]],
                                   dim=1) # [num_nodes, 2, C * 2]
        scores_features = features_list.view(-1, features_list.shape[-1]) # [num_nodes * 2, C * 2]
        fusion_scores = (left_soft_label * right_soft_label).sum(dim=1) # [num_nodes]
        labels = fusion_scores[:, None].repeat(1, 2).view(-1) # [num_nodes * 2]
        node_id_list = list(range(num_batch, num_batch + len(tree_connection))) # [num_nodes]

        # inverse range to realize traverse top-down
        num_all_nodes = len(node_id_list)
        scores_features = scores_features[range(-1, -(num_all_nodes*2+1), -1)] # [num_nodes * 2, C]
        labels = labels[range(-1, -(num_all_nodes*2+1), -1)] # [num_nodes * 2]
        nodes = torch.Tensor(node_id_list).to(scores_features.device)[range(-1, -(num_all_nodes+1), -1)] # [num_nodes]
        scores_features_list.append(scores_features)
        labels_list.append(labels)
        nodes_list.append(nodes)

        tree = Tree()
        tree.create_node(num_nodes, num_nodes, data=node_list[num_nodes]) # root node
        for connection in tree_connection[::-1]:
            c0, c1 = connection
            tree.create_node(c0, c0, parent=num_nodes, data=node_list[c0])
            tree.create_node(c1, c1, parent=num_nodes, data=node_list[c1])
            num_nodes -= 1
        
        tree_list.append(tree)

    return hierarchical_tree_list, tree_list, scores_features_list, labels_list, nodes_list


def get_fusion_property(properties: torch.Tensor,
                        count: torch.Tensor,
                        leaves: torch.Tensor,
                        ids: torch.Tensor,
                        nodes_count: torch.Tensor) -> torch.Tensor:
    r"""get the fused properties of fusion for nodes(HNIR)

    Args:
        properties (torch.Tensor, [N, C]): properties to be fused
        count (torch.Tensor, [num_leaves]): points number of each leaf
        leaves (torch.Tensor, [num_leaves]): leaf ids to label properties
        ids (torch.Tensor, [num_leaves]): node ids of each leaf
        nodes_count (torch.Tensor, [num_nodes]): points number of each node

    Returns:
        torch.Tensor: [description]
    """
    num_leaves = leaves.shape[0]
    properties = properties[leaves].view(num_leaves, -1) # [num_leaves, C]
    property_gain = properties * count[leaves].view(num_leaves, 1) # [num_leaves, C]
    properties = scatter_add(property_gain, ids, dim=0) # [num_nodes, C]
    properties = properties / nodes_count.view(-1, 1) # [num_nodes, C]
    return properties


def align_superpoint_label(labels: torch.Tensor,
                           superpoint: torch.Tensor,
                           num_label: int=20,
                           ignore_label: int=-100):
    r"""refine semantic segmentation by superpoint

    Args:
        labels (torch.Tensor, [N]): semantic label of points
        superpoint (torch.Tensor, [N]): superpoint cluster id of points
        num_label (int): number of valid label categories
        ignore_label (int): the ignore label id

    Returns:
        label: (torch.Tensor, [num_superpoint]): superpoint's label
        label_scores: (torch.Tensor, [num_superpoint, num_label + 1]): superpoint's label scores
    """
    row = superpoint.cpu().numpy()      # [N] superpoint has been compression
    col = labels.cpu().numpy()          # [N]
    col[col < 0] = num_label            # -100 = num_label  num_label = max_inst_id+1
    data = np.ones(len(superpoint))     # [N]
    shape = (len(np.unique(row)), num_label + 1)
    label_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_label + 1]
    """
    coo_matrix((data, (i, j)), [shape=(M, N)])
        to construct from three arrays:
            data[:] the entries of the matrix, in any order
            i[:] the row indices of the matrix entries
            j[:] the column indices of the matrix entries
        Where A[i[k], j[k]] = data[k]. When shape is not specified, it is inferred from the index arrays

    scipy: .todense() return matrix   .toarray() return ndarray

    >>> row  = np.array([0,0,3,1,1,0])
    >>> col  = np.array([0,0,3,1,1,2])
    >>> data = np.array([4,2,5,7,1,9])
    >>> coo_matrix((data,(row,col)), shape=(4,4)).todense()
    matrix([[6, 0, 9, 0],
            [0, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 5]])
    """
    label = torch.Tensor(np.argmax(label_map, axis=1)).long().to(labels.device)     # [num_superpoint]
    label[label == num_label] = ignore_label # ignore_label
    label_scores = torch.Tensor(label_map / label_map.sum(axis=1)[:, None]).to(labels.device) # [num_superpoint, num_label + 1]

    return label, label_scores


def voting_semantic_segmentation(semantic_preds: torch.Tensor,
                                 superpoint: torch.Tensor,
                                 num_semantic: int=20):
    r"""get semantic segmentation by superpoint voting

    Args:
        semantic_preds (torch.Tensor, [N]): semantic label of points
        superpoint (torch.Tensor, [N]): superpoint cluster id of points
        num_semantic (int): the number of semantic labels

    Returns:
        replace_semantic: (torch.Tensor, [N]): refine semantic label of points
    """
    _, row = np.unique(superpoint.cpu().numpy(), return_inverse=True)
    col = semantic_preds.cpu().numpy()
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_semantic)
    semantic_map = coo_matrix((data, (row, col)), shape=shape).toarray()                    # [num_superpoint, num_semantic]
    semantic_map = torch.Tensor(np.argmax(semantic_map, axis=1)).to(semantic_preds.device)  # [num_superpoint]
    replace_semantic = semantic_map[torch.Tensor(row).to(semantic_preds.device).long()]

    return replace_semantic


def traversal_cluster(tree: Tree,
                      nodes: List[int],
                      fusion_labels: List[bool]):
    r"""
    get the cluster result by top-down bfs traversing hierachical tree

    Args:
        tree (treelib.Tree): [description]
        nodes (torch.Tensor, [num_nodes]): [description]
        scores (torch.Tensor, [num_nodes * 2]): [description]

    Returns:
        List[List[List[int]], List[int]], list of cluster superpoint id and list of node id
    """
    queue = [tree.root]
    
    cluster_list = []
    node_id_list = []
    # refine_labels = []
    nodes_ids = []
    leaves_ids = []
    nodes_soft_label = []
    leaves_soft_labels = []
    while (len(queue) > 0):
        # get aim point id from queue
        node_id = queue.pop(0)
        idx = nodes.index(node_id)
        if fusion_labels[idx]:
            leaves = [l.tag for l in tree.leaves(node_id)]
            cluster_list.append(leaves)
            node_id_list.append(node_id)
            nodes_ids.extend([node_id] * len(leaves))
            leaves_ids.extend(leaves)
            nodes_soft_label.extend([tree.get_node(node_id).data.soft_label] * len(leaves))
            leaves_soft_labels.extend([tree.get_node(l).data.soft_label for l in leaves])
        else:
            child = tree.children(node_id)
            for c in child:
                nid = c.tag
                # child
                if len(tree.children(nid)) > 0:
                    queue.append(nid)

    try:
        nodes_soft_label = torch.stack(nodes_soft_label)
        leaves_soft_labels = torch.stack(leaves_soft_labels)
        refine_labels = (nodes_soft_label * leaves_soft_labels).sum(1)
    except:
        cluster_list = None
        node_id_list = None
        refine_labels = None

    return cluster_list, node_id_list, refine_labels


def build_superpoint_clique(tree: Tree,
                           node_id_list: List[List[int]]):
    r"""build the superpoint clique for refinement

    Args:
        tree (Tree): input sstnet
        node_id_list (List[List[int]]): node ids of each proposal
    """
    num_leaves = len(tree.leaves(tree.root))
    num_graph_nodes = num_leaves + len(node_id_list)
    # self connection
    dense_matrix = torch.eye(num_graph_nodes).float()
    # conver the sub_tree as graph
    for idx, node_id in enumerate(node_id_list):
        leaves = tree.leaves(node_id)
        root_id = num_leaves + idx
        for leaf in leaves:
            nid = leaf.tag
            dense_matrix[nid, root_id] = 1
            dense_matrix[root_id, nid] = 1
    dense_matrix = F.normalize(dense_matrix, p=1, dim=-2)
    
    # construct sparse matrix to represent graph connection
    indices = torch.where(dense_matrix > 0)
    i = torch.stack(indices)
    v = dense_matrix[indices]
    adjancy_matrix = torch.sparse.FloatTensor(i, v, torch.Size([num_graph_nodes, num_graph_nodes]))
    return adjancy_matrix


def get_proposals_idx(superpoint: torch.Tensor, cluster_list: List[List[int]]):
    r"""
    get proposals idx(mask) from superpoint clusters

    Args:
        superpoint (torch.Tensor): superpoint ids
        cluster_list (List[List[int]]): List of cluster ids

    Returns:
        proposals_idx
    """
    superpoint_np = superpoint.cpu().numpy()
    proposals_idx_list = []
    cluster_id = 0
    for cluster in cluster_list:
        proposals_idx = np.where(np.isin(superpoint_np, cluster))[0]
        clusters_id = np.ones_like(proposals_idx) * cluster_id
        proposals_idx = np.stack([clusters_id, proposals_idx], axis=1)
        if len(proposals_idx) < 50:
            continue
        proposals_idx_list.append(proposals_idx)
        cluster_id += 1
    proposals_idx = np.concatenate(proposals_idx_list)
    proposals_idx = torch.from_numpy(proposals_idx)

    return proposals_idx

