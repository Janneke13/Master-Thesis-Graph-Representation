import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.transforms import RandomLinkSplit
import torch.nn.functional as F
import os
import reading_data
import utils
import csv

"""
This file contains all code to run the GAE model.
It contains several parameters with which the parameters of the model itself can be changed.
NOTE: for the classification tasks, the training, validation, and test sets are all already created. As the GAE works
on an edge-level, the edge splitting function of PyTorch Geometric is used.

Partly based on: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py.
"""


class GCN_encoder(torch.nn.Module):
    """
    Uses GCN layers to encode the nodes.
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super().__init__()
        self.layer1 = GCNConv(input_nodes, hidden_nodes)
        self.layer2 = GCNConv(hidden_nodes, output_nodes)

    def forward(self, X, A):
        h1 = self.layer1(X, A)
        h1_activated = F.relu(h1)
        h2 = self.layer2(h1_activated, A)
        return h2


def run_gae_model(dataset_filename, literal_representation, hidden_nodes, output_nodes, optimizer, learning_rate,
                  weight_decay, nr_epochs, seed, test=True, record_results=False):
    """
    Runs the GAE model for a certain dataset and literal representation, and records the results.
    :param dataset_filename: The filename of the dataset; where it is stored.
    :param literal_representation: The type of literal representation for the adjacency matrix.
    :param hidden_nodes: The number of hidden nodes of the GAE.
    :param output_nodes: The number of output nodes of the GAE.
    :param optimizer: The optimizer used for the GAE.
    :param learning_rate: The learning rate used for the GAE.
    :param weight_decay: The weight decay used for the optimizer of the GAE.
    :param nr_epochs: The number of epochs used for the GAE.
    :param seed: The random seed used for the GAE
    :param test: A Boolean stating whether the test data is the validation or the test data.
    :param record_results: A Boolean stating whether results need to be stored in CSV files or not.
    :return:
    """

    # assert whether aspects of the input data are properly defined
    assert optimizer in ["adam", "sgd", "adagrad"], "Optimizer must be one of: 'adam', 'sgd', 'adagrad'."

    # set the seed
    torch_geometric.seed.seed_everything(seed)

    # put everything into place to record the results:
    if record_results:
        utils.create_results_folders("GAE")

        # create new file (that does not exist yet)
        current_test = 1
        while os.path.exists("results/GAE/" + str(current_test) + ".csv"):
            current_test += 1

        # make a file to record the results
        file_results = open("results/GAE/" + str(current_test) + ".csv", "w")
        writer_results = csv.writer(file_results)

        # create a header
        header = ["Epoch", "ReconLossTrain", "ReconLossTest", "AUCTest", "APTest"]
        writer_results.writerow(header)

        # save the configuration --> so it can be checked later!
        file_config = open("results/GAE/" + str(current_test) + "_config.csv", "w")
        writer_config = csv.writer(file_config)

        # create a header for the configuration
        header_config = ["Dataset", "Literal_Rep", "Number_Epochs", "Hidden_Nodes", "Output_Nodes", "Optimizer",
                         "Learning_Rate", "Weight_Decay", "Random_Seed", "Test"]

        # write in all the configuration data
        config = [dataset_filename, literal_representation, nr_epochs, hidden_nodes, output_nodes, optimizer,
                  learning_rate, weight_decay, seed, test]

        # write the configuration and close it --> file is just for recording of results!
        writer_config.writerow(header_config)
        writer_config.writerow(config)
        file_config.close()

    # ---- MODEL ------

    # create the adjacency matrix
    adjacency_matrix, mapping_index_to_node, mapping_entity_to_index = reading_data.create_adjacency_matrix_nt(
        dataset_filename, literal_representation=literal_representation, sparse=True)

    # record the number of nodes
    number_nodes = adjacency_matrix.size()[0]

    # create the one-hot feature matrix:
    feature_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([list(range(number_nodes)), list(range(number_nodes))]), values=torch.ones(number_nodes),
        size=(number_nodes, number_nodes))

    # create a data object --> with the already created adjacency matrix
    data = Data(x=feature_matrix, edge_index=adjacency_matrix.coalesce().indices(), num_nodes=number_nodes)

    # split the edges into a training, validation, and test sets
    # set up a link split function --> have to split the labels into train, test and validation
    transformation = RandomLinkSplit(split_labels=True, add_negative_train_samples=False)

    # split the data into a train, validation and test set
    train_data, val_data, test_data = transformation(data)

    # create an encoder with the number of input nodes, hidden nodes, and output nodes as defined
    encoder = GCN_encoder(data.num_features, hidden_nodes, output_nodes)

    # create the GAE model --> if decoder is not defined, it is the inner product decoder as given by the original paper
    model = GAE(encoder)

    # there are three types of optimizer that can be put in (for now):
    if optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # run the training loop:
    # dataset_filename, literal_representation, hidden_nodes, output_nodes, optimizer, learning_rate,
    #                   weight_decay, nr_epochs, seed, test=True, record_results=False):
    for epoch in range(nr_epochs):
        # set to training mode, put the grads to zero:
        model.train()
        optim.zero_grad()

        # encode everything in latent space using the encoder:
        latent_space = model.encode(train_data.x, train_data.edge_index)

        # calculate the reconstruction loss and perform the backwards operation
        reconstruction_loss = model.recon_loss(latent_space, train_data.pos_edge_label_index)
        reconstruction_loss.backward()

        # update the parameters
        optim.step()

        # TODO: add additional improvement steps here, such as gradient clipping

        # now, evaluate the model and save the information:
        model.eval()

        # when test mode is used
        if test:
            latent_space_test = model.encode(test_data.x, test_data.edge_index)
            reconstruction_loss_test = model.recon_loss(latent_space_test, test_data.pos_edge_label_index,
                                                        test_data.neg_edge_label_index)
            AUC_test, average_precision_test = model.test(latent_space_test, test_data.pos_edge_label_index,
                                                          test_data.neg_edge_label_index)
        # when validation mode is used
        else:
            latent_space_test = model.encode(val_data.x, val_data.edge_index)
            reconstruction_loss_test = model.recon_loss(latent_space_test, val_data.pos_edge_label_index,
                                                        val_data.neg_edge_label_index)
            AUC_test, average_precision_test = model.test(latent_space_test, val_data.pos_edge_label_index,
                                                          val_data.neg_edge_label_index)

        # if the results are recorded, then add a row to the csv!
        if record_results:
            row = [epoch + 1, reconstruction_loss.item(), reconstruction_loss_test.item(), AUC_test.item(),
                   average_precision_test.item()]
            writer_results.writerow(row)
        # otherwise, print them
        else:
            print("Epoch: ", epoch + 1, ", Rec_Loss_Train: ", reconstruction_loss.item(), ", Rec_Loss_Test: ",
                  reconstruction_loss_test.item(), ", AUC_test: ", AUC_test.item(), ", AP_test: ",
                  average_precision_test.item())

    # TODO: add clustering(!), or more experiments with the GAE

    # close the file if the results were recorded:
    if record_results:
        # close the file
        file_results.close()
