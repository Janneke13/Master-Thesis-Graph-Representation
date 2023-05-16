import csv
import os
import torch
from torch_geometric.nn import GAE
import utils
from GCN_model import GCN
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
import string
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
from sklearn.metrics.cluster import adjusted_rand_score

"""
Uses the GAE to encode everything, will end on 2 coordinates. Then, clusters this data and creates a scatter plot of it.

Creates and saves three scatter plots --> using random_state 1, 2, and 3 --> to check whether the clustering is stable. 
Saves a plot of the rec loss and the contingency tables of the clustering as well.
"""


def run_gae_model(data_name, data, hidden_nodes, optimizer, learning_rate, weight_decay, nr_epochs, label_map, seed,
                  literal_mapping):
    """
    Runs the GAE model and records the results - performs the entire training loop.
    :param data_name: The name of the dataset (for documentation purposes).
    :param data: The data object including the edges and the labels.
    :param hidden_nodes: The number of hidden nodes of the GAE.
    :param optimizer: The optimizer used for the GAE.
    :param learning_rate: The learning rate used for the GAE.
    :param weight_decay: The weight decay used for the optimizer of the GAE.
    :param nr_epochs: The number of epochs used for the GAE.
    :param label_map: The mapping of the labels with label:id.
    :param seed: The seed used to initialize everything (set before entering this method), for recording purposes only.
    :param literal_mapping: The type of literal mapping used, for recording purpose only.
    :return: A dictionary with the results and the model.
    """

    # assert whether aspects of the input data are properly defined
    assert optimizer in ["adam", "sgd", "adagrad"], "Optimizer must be one of: 'adam', 'sgd', 'adagrad'."

    # ----- CREATE FILES AND STORE CONFIGURATION --------
    # create new folder (that does not exist yet)
    current_test = 1
    while os.path.exists("results/GAE/" + data_name + "_" + str(current_test)):
        current_test += 1

    utils.create_results_folders("GAE", data_name + "_" + str(current_test))

    # make a file to record the reconstruction loss
    file_results = open("results/GAE/" + data_name + "_" + str(current_test) + "/results.csv", "w")
    writer_results = csv.writer(file_results)

    # create a header
    header = ["Epoch", "ReconLossTrain"]
    writer_results.writerow(header)

    # make a file to record the final results (including those of clustering) ------
    file_results_final = open("results/GAE/" + data_name + "_" + str(current_test) + "/final_results.csv", "w")
    writer_final_results = csv.writer(file_results_final)

    # write the header --> bic scores for the clustering, rand12 stands for the rand score of cluster 1 and cluster 2
    header = ["ReconLossFinal", "BIC_1", "BIC_2", "BIC_3", "Rand12", "Rand23", "Rand13"]
    writer_final_results.writerow(header)

    # save the configuration --> so it can be checked later!
    file_config = open("results/GAE/" + data_name + "_" + str(current_test) + "/config.csv", "w")
    writer_config = csv.writer(file_config)

    # create a header for the configuration
    header_config = ["Number_Epochs", "Hidden_Nodes", "Optimizer", "Learning_Rate", "Weight_Decay", "Seed", "Literal "
                                                                                                            "Mapping"]

    # write in all the configuration data
    config = [nr_epochs, hidden_nodes, optimizer, learning_rate, weight_decay, seed, literal_mapping]

    # write the configuration and close it --> file is just for recording of configuration!
    writer_config.writerow(header_config)
    writer_config.writerow(config)
    file_config.close()

    # ---- MODEL ------
    # using two coordinated, so use this as the number of output nodes
    output_nodes = 2

    # create an encoder with the number of input nodes, hidden nodes, and output nodes as defined
    encoder = GCN(data.num_features, hidden_nodes, output_nodes)

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

    # check whether cuda can be used and if so, set it as the device!
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    data.to(device)

    # run the training loop:
    for epoch in range(nr_epochs):
        # set to training mode, put the grads to zero:
        model.train()
        optim.zero_grad()

        # encode everything in latent space using the encoder:
        latent_space = model.encode(data.x, data.edge_index)

        # calculate the reconstruction loss and perform the backwards operation
        reconstruction_loss = model.recon_loss(latent_space, data.edge_index)
        reconstruction_loss.backward()

        # update the parameters
        optim.step()

        # now, evaluate the model and save the information:
        model.eval()

        # add it to the lists as well:
        loss_list_tr.append(reconstruction_loss.item())

        row = [epoch, reconstruction_loss.item()]
        writer_results.writerow(row)

    # close the file
    file_results.close()

    # make a dictionary of the results and the model, so they can be used later if needed
    results_dict = {"model": model, "loss_list_train": loss_list_tr}

    # set it to evaluate, and let the model encode the graph again
    model.eval()
    latent_space_final = model.encode(data.x, data.edge_index)

    # put the coordinates in a dataframe:
    coordinates = pd.DataFrame(latent_space_final.detach().numpy())

    # put the data to the cpu, as it needs to be used for statistics now!
    data = data.cpu()

    # add the labels and filter the dataframe to only get the labelled entities
    coordinates['labels'] = data.y.int()

    # set the new labeled coordinates
    coordinates_labelled = coordinates[coordinates['labels'] != 0]

    # find out how many clusters there need to be
    nr_clusters = len(coordinates_labelled['labels'].unique())

    # reverse the label mapping so it can be used later
    mapping_labels = {value: key.split('/')[-1] for key, value in label_map.items()}

    # add the label names
    coordinates_labelled['label_names'] = coordinates_labelled['labels'].map(mapping_labels)

    # store for the final results
    bic_scores = []
    clustering = []

    # cluster the output, do this three times, with different random states (to check whether it is stable)
    for random_seed in [1, 2, 3]:
        plt.figure()

        # cluster the output
        gmm = GaussianMixture(n_components=nr_clusters, random_state=random_seed)
        gmm.fit(coordinates_labelled[[0, 1]])
        predictions = gmm.predict(coordinates_labelled[[0, 1]])

        # append the bic score and the current clusters
        bic_scores.append(gmm.bic(coordinates_labelled[[0, 1]]))
        clustering.append(predictions)

        # set the predictions in there!
        coordinates_labelled['predictions'] = predictions

        # print the contingency matrix
        con_table = pd.crosstab(coordinates_labelled['label_names'], predictions)
        con_table.to_csv("results/GAE/" + data_name + "_" + str(current_test) + "/contigency_"
                         + str(random_seed) + ".csv")

        # set colors
        colors = get_cmap("Set1").colors

        # set markers
        markers = list(string.ascii_uppercase)

        # so we have the correct order for the legend later on
        handles = []
        handles_colors = []

        # -1 stands for unlabelled
        for i in range(0, nr_clusters):
            # in the clustering from sklearn, it starts with 0 and goes up
            for j in range(nr_clusters):
                # find the specific slice that contains the ones with a specific marker and color
                current_pred_class = coordinates_labelled[(coordinates_labelled['predictions'] == j)
                                                          & (coordinates_labelled['labels'] == i)]

                # color as the true label, form as the predicted label
                plt.scatter(current_pred_class[0], current_pred_class[1], marker='$' + markers[j] + '$',
                            color=colors[i - 1])

                if mapping_labels[i] not in handles:
                    handles.append(mapping_labels[i])
                    handles_colors.append(mpatches.Patch(color=colors[i], label=mapping_labels[i]))

        plt.legend(handles=handles_colors)

        # save the figure
        plt.savefig("results/GAE/" + data_name + "_" + str(current_test) + "/clusterplot_" + str(random_seed) + ".jpeg",
                    dpi=300)

    # write the final results:
    rand_in12 = adjusted_rand_score(clustering[0], clustering[1])
    rand_in23 = adjusted_rand_score(clustering[1], clustering[2])
    rand_in13 = adjusted_rand_score(clustering[0], clustering[2])

    fin_res = [loss_list_tr[-1], bic_scores[0], bic_scores[1], bic_scores[2], rand_in12, rand_in23, rand_in13]
    writer_final_results.writerow(fin_res)

    # close the file of the final results
    file_results_final.close()

    plt.figure()
    plt.plot(loss_list_tr)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.savefig("results/GAE/" + data_name + "_" + str(current_test) + "/rec_loss" + ".jpeg", dpi=300)

    return results_dict
