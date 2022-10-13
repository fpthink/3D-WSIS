"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import igraph
import torch
from collections import defaultdict
import numpy as np
    
class GraphConvInfo(object):          
    """ Holds information about the structure of graph(s) in a vectorized form useful to `GraphConvModule`. 
    
    We assume that the node feature tensor (given to `GraphConvModule` as input) is ordered by igraph vertex id, e.g. the fifth row corresponds to vertex with id=4. Batch processing is realized by concatenating all graphs into a large graph of disconnected components (and all node feature tensors into a large tensor).

    The class requires problem-specific `edge_feat_func` function, which receives dict of edge attributes and returns Tensor of edge features and LongTensor of inverse indices if edge compaction was performed (less unique edge features than edges so some may be reused).
    """

    def __init__(self, *args, **kwargs):
        self._idxn = None           # indices into input tensor of convolution (node features)s
        self._idxe = None           # indices into edge features tensor (or None if it would be linear, i.e. no compaction)
        self._degrees = None        # in-degrees of output nodes (slices _idxn and _idxe)
        self._degrees_gpu = None
        self._edgefeats = None      # edge features tensor (to be processed by feature-generating network)
        self.edges_for_ext = None
        if len(args)>0 or len(kwargs)>0:
            self.set_batch(*args, **kwargs)
      
    def set_batch(self, graphs, edge_feat_func):
        """ Creates a representation of a given batch of graphs.
        
        Parameters:
        graphs: single graph or a list/tuple of graphs.
        edge_feat_func: see class description.
        """
        
        graphs = graphs if isinstance(graphs,(list,tuple)) else [graphs]
        p = 0   # base index for processsing all superpoints
        idxn = [] # list, containing list consists of reordered idx of source vertexes of edges
        degrees = []    # the number of edges pointing towards the vertex
        edge_indexes = []   # edge index
        edgeattrs = defaultdict(list)
        edges_for_ext = []  # rearrange the edge in the batch-level graph
                
        for i, G in enumerate(graphs):  # rearrange the index of node in the whole batch
            E = np.array(G.get_edgelist()) #   source, target
            # print('E: {}'.format(E.shape))

            idx = E[:,1].argsort() # sort by target, the value is the index that from small to large
            # 如果没有Edge，那么会报错误：IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            
            idxn.append(p + E[idx,0]) # reordered source vertex
            edges_for_ext.append(p + E)
            # https://igraph.org/python/doc/tutorial/tutorial.html#setting-and-retrieving-attributes
            # G.vs: the sequence of all vertices
            # G.es: the sequence of all edges
            edgeseq = G.es[idx.tolist()] # igraph.EdgeSeq, list, N_edges, 13-dim    seq for sequence
            # for e in edgeseq:
            #     print('e: {}, {}, {}'.format(e['f'], e['f'].shape, type(e['f'])))
            #     exit()

            for a in G.es.attributes(): # for keys of edge attributes, there is only 'f' in this code
                edgeattrs[a] += edgeseq.get_attribute_values(a)
            degrees += G.indegree(G.vs, loops=True) # the number of edges pointing towards the vertex
            edge_indexes.append(np.asarray(p + E[idx]))
            p += G.vcount()
              
        self._edgefeats, self._idxe = edge_feat_func(edgeattrs)

        # 读取edge label
        is1ins = np.asarray(edgeattrs['is1ins']) ####
        self.is1ins_labels = torch.from_numpy(is1ins).to(torch.float32) ####
        
        self._idxn = torch.LongTensor(np.concatenate(idxn))
        if self._idxe is not None:
            assert self._idxe.numel() == self._idxn.numel()     # .numel() return the number of elements
            
        self._degrees = torch.LongTensor(degrees)
        self._degrees_gpu = None            
        self.edges_for_ext = torch.from_numpy(np.concatenate(edges_for_ext, 0)).long()  # edge information (u, v)

        self._edge_indexes = torch.LongTensor(np.concatenate(edge_indexes).T)      # edge indexes after sorted 
        
    def cuda(self):
        self._idxn = self._idxn.cuda()
        if self._idxe is not None: self._idxe = self._idxe.cuda()
        self._degrees_gpu = self._degrees.cuda()
        self._edgefeats = self._edgefeats.cuda()      
        self._edge_indexes = self._edge_indexes.cuda()  
        
    def get_buffers(self):
        """ Provides data to `GraphConvModule`.
        """
        return self._idxn, self._idxe, self._degrees, self._degrees_gpu, self._edgefeats

    def get_pyg_buffers(self):
        """ Provides data to `GraphConvModule`.
        """
        return self._edge_indexes
