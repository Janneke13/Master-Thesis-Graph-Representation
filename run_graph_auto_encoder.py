import csv
import os
import torch
from torch_geometric.nn import GAE
import utils
from GCN_model import GCN

"""
Similar to the run_classification_model.py file, but the GAE uses a different way of splitting the data. Also, its
output is different. Therefore, a different class is created. 

NOTE: for the classification tasks, the training, validation, and test sets are all already created. As the GAE works
on an edge-level, the edge splitting function of PyTorch Geometric is used beforehand, which also creates the false
edges.

Partly based on: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py.
"""


def run_gae_model(train_data, val_data, test_data, hidden_nodes, output_nodes, optimizer, learning_rate,
                  weight_decay, nr_epochs, test=True, record_results=False, path_folder=None):
    """
    Runs the GAE model and records the results - performs the entire training loop.
    :param train_data: The pre-created training edges.
    :param val_data: The pre-created validation edges.
    :param test_data: The pre-created test edges.
    :param hidden_nodes: The number of hidden nodes of the GAE.
    :param output_nodes: The number of output nodes of the GAE.
    :param optimizer: The optimizer used for the GAE.
    :param learning_rate: The learning rate used for the GAE.
    :param weight_decay: The weight decay used for the optimizer of the GAE.
    :param nr_epochs: The number of epochs used for the GAE.
    :param test: A Boolean stating whether the test data is the validation or the test data.
    :param record_results: A Boolean stating whether results need to be stored in CSV files or not.
    :param path_folder: A path to the folder where the results are stored.
    :return: A dictionary with the results and the model.
    """

    # assert whether aspects of the input data are properly defined
    assert optimizer in ["adam", "sgd", "adagrad"], "Optimizer must be one of: 'adam', 'sgd', 'adagrad'."

    # put everything into place to record the results:
    if record_results:
        utils.create_results_folders("GAE", path_folder)

        # create new file (that does not exist yet)
        current_test = 1
        while os.path.exists("results/GAE/" + path_folder + "/" + str(current_test) + ".csv"):
            current_test += 1

        # make a file to record the results
        file_results = open("results/GAE/" + path_folder + "/" + str(current_test) + ".csv", "w")
        writer_results = csv.writer(file_results)

        # create a header
        header = ["Epoch", "ReconLossTrain", "ReconLossTest", "AUCTest", "APTest"]
        writer_results.writerow(header)

        # save the configuration --> so it can be checked later!
        file_config = open("results/GAE/" + path_folder + "/" + str(current_test) + "_config.csv", "w")
        writer_config = csv.writer(file_config)

        # create a header for the configuration
        header_config = ["Number_Epochs", "Hidden_Nodes", "Output_Nodes", "Optimizer",
                         "Learning_Rate", "Weight_Decay", "Test"]

        # write in all the configuration data
        config = [nr_epochs, hidden_nodes, output_nodes, optimizer,
                  learning_rate, weight_decay, test]

        # write the configuration and close it --> file is just for recording of results!
        writer_config.writerow(header_config)
        writer_config.writerow(config)
        file_config.close()

    # ---- MODEL ------
    # create an encoder with the number of input nodes, hidden nodes, and output nodes as defined
    encoder = GCN(train_data.num_features, hidden_nodes, output_nodes)

    # create the GAE model --> if decoder is not defined, it is the inner product decoder as given by the original paper
    model = GAE(encoder)

    # there are three types of optimizer that can be put in (for now):
    if optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # create loss lists etc., to use later on
    loss_list_tr = list()
    loss_list_te = list()
    ap_list = list()
    auc_list = list()

    # run the training loop:
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

        # add it to the lists as well:
        loss_list_tr.append(reconstruction_loss.item())
        loss_list_te.append(reconstruction_loss_test.item())
        ap_list.append(average_precision_test.item())
        auc_list.append(AUC_test.item())

    # close the file if the results were recorded:
    if record_results:
        # close the file
        file_results.close()

    # make a dictionary of the results and the model, so they can be used later if needed
    results_dict = {"model": model, "loss_list_train": loss_list_tr, "loss_list_test": loss_list_te,
                    "ap_list": ap_list, "auc_list": auc_list}

    return results_dict
