import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

"""
File that contains the code to create a GCN, as well as run it using a certain dataset.
Results can be printed or stored, and are also returned by the function.
"""


class GCN(torch.nn.Module):
    """
    A basic two-layer Graph Convolutional Network. From the paper: <add reference>
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        """
        Initializes the GCN.
        :param input_nodes: The number of input nodes.
        :param hidden_nodes: The number of hidden nodes.
        :param output_nodes: The number of output nodes.
        """
        super().__init__()
        self.layer1 = GCNConv(input_nodes, hidden_nodes)
        self.layer2 = GCNConv(hidden_nodes, output_nodes)

    def forward(self, X, A):
        """
        Performs one forward pass of the GCN.
        :param X: The feature matrix (dense).
        :param A: The edge indices (adjacency matrix).
        :return: The last, not-activated layer output. This represents the embeddings.
        """
        h1 = self.layer1(X, A)
        h1_activated = F.relu(h1)
        h2 = self.layer2(h1_activated, A)
        return h2
