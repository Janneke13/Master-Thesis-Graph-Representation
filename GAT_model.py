import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.nn import Dropout


class GAT(torch.nn.Module):
    """
    Implementation of the Graph Attention Network.
    Based on paper <add reference to paper>
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, nr_heads, dropout):
        """
        Intializes the GAT model, using the GATConv layers of PyTorch Geometric.
        :param input_nodes: The number of input nodes of the GAT model.
        :param hidden_nodes: The number of hidden nodes of the GAT model.
        :param output_nodes: The number of output nodes of the GAT model.
        :param nr_heads: The number of heads of the first GAT layer.
        :param dropout: The probability of dropout during the training phase. Note that the dropout probability is the
        same for every layer, for both the attention coefficients and the input.
        """
        super().__init__()
        # dropout is applied to the GAT convolutional layer as well, as described in the paper
        self.layer1 = GATConv(input_nodes, hidden_nodes, heads=nr_heads, dropout=dropout)

        # need to put heads to 1, to be able to get output
        self.layer2 = GATConv(nr_heads * hidden_nodes, output_nodes, heads=1, dropout=dropout)

        # use the same probability for everything
        self.dropout = Dropout(p=dropout)

    def forward(self, X, A):
        """
        Performs one forward pass of the initialized GAT model.
        :param X: The feature matrix, needs to be dense.
        :param A: The adjacency matrix in COO format.
        :return: The embeddings the GAT model produces.
        """
        # in the paper, they use dropout on both of the layers' input while training
        X = self.dropout(X)
        h1 = self.layer1(X, A)

        # in the paper they use an exponential linear unit, which is used here as well:
        h1_activated = F.elu(h1)
        h1_activated = self.dropout(h1_activated)

        h2 = self.layer2(h1_activated, A)
        # cross entropy loss used, therefore a softmax is not needed.
        return h2


def run_gat_model():
    pass
