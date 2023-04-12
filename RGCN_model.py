import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

"""
File containing the implementation of the R-GCN model.
"""


class RGCN(torch.nn.Module):
    """
    A class to create a basic two-layer Relational GCN.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, num_rel):
        """
        Initialized the R-GCN.
        :param input_nodes: The number of input nodes (the features).
        :param hidden_nodes: The number of hidden nodes.
        :param output_nodes: The number of output nodes.
        :param num_rel: The number of relations the R-GCN needs to take into account.
        """
        super().__init__()
        self.layer1 = RGCNConv(input_nodes, hidden_nodes, num_relations=num_rel)
        self.layer2 = RGCNConv(hidden_nodes, output_nodes, num_relations=num_rel)

    def forward(self, X, A, edge_type):
        """
        Performs one forward step of the R-GCN.
        :param X: The (dense) tensor containing the features.
        :param A: The adjacency matrix, in sparse COO format.
        :param edge_type: A tensor of thed edge types of A.
        :return: The (non-activated) output of the second RGCN layer.
        """
        h1 = self.layer1(X, A, edge_type)
        h1_activated = F.relu(h1)
        h2 = self.layer2(h1_activated, A, edge_type)
        return h2
