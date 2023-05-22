from run_classification_models import run_classification_model
import torch
from torch_geometric.data import Data
import reading_data
import torch_geometric
from run_graph_auto_encoder import run_gae_model
import argparse

"""
Runs experiments in loops --> only reads in data once.
Used for hyperparameter optimization for convenience, as it speeds up the process a lot.
This file is specifically for the LEARNING RATE and WEIGHT DECAY--> will test the following: 0.1, 0.01, 0.001, 0.005 for
LR, and 0.1, 0.01, 0.001 for weight decay (did some ad-hoc tests, and these seemed decent).
Runs it for seeds 0, 1, 2 and for all literal mappings!

It changes one hyperparameter (learning rate) while keeping the other ones constant! The defaults are used here!
"""

# use a parser so the command line can be used to set up everything (useful when used with DAS-5)
parser = argparse.ArgumentParser()

# basic arguments - such as the dataset and the random seed added
parser.add_argument("--dataset", type=str, default='mutag', help="The used dataset, can be AIFB+, MUTAG or DMG777k.")

# arguments for the model itself
parser.add_argument("--model", type=str, default="RGCN", help="The model used.")
parser.add_argument("--hidden_nodes", type=int, default=16, help="The amount of hidden nodes in the GNN.")
parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer used to train the GNN.")
parser.add_argument("--num_epochs", type=int, default=250, help="Number of epochs used for training.")

# arguments specifically for the GAT (ignored when GAT is not used)
parser.add_argument("--dropout_gat", type=float, default=0.6, help="Dropout probability used for GAT.")
parser.add_argument("--nr_attention_heads", type=int, default=8, help="Number of attention heads used by the GAT.")

# read all the arguments from the command line
arguments = parser.parse_args()

# set the parameters
dataset = arguments.dataset
model = arguments.model
hidden_nodes = arguments.hidden_nodes
optimizer = arguments.optimizer
num_epochs = arguments.num_epochs
dropout_gat = arguments.dropout_gat
nr_attention_heads = arguments.nr_attention_heads

# read in the data and create data objects:
# set the correct path:
if dataset == 'aifb':
    path_data = "data/aifb/gz_files/aifb+.nt.gz"
    path_train = "data/aifb/gz_files/aifb+_train_set.nt.gz"
    path_valid = "data/aifb/gz_files/aifb+_valid_set.nt.gz"
    path_test = "data/aifb/gz_files/aifb+_test_set.nt.gz"
elif dataset == 'mutag':
    path_data = "data/mutag/gz_files/mutag.nt.gz"
    path_train = "data/mutag/gz_files/mutag_train_set.nt.gz"
    path_valid = "data/mutag/gz_files/mutag_valid_set.nt.gz"
    path_test = "data/mutag/gz_files/mutag_test_set.nt.gz"
else:  # dmg777k
    path_data = "data/dmg777k/gz_files/dmg777k_stripped.nt.gz"
    path_train = "data/dmg777k/gz_files/dmg777k_train_set.nt.gz"
    path_valid = "data/dmg777k/gz_files/dmg777k_valid_set.nt.gz"
    path_test = "data/dmg777k/gz_files/dmg777k_test_set.nt.gz"


for literal_map in ["filtered", "all-to-one", "collapsed", "separate"]:

    # read in the data, set relational to True --> can be ignored if not needed, and does not take longer
    adjacency_matrix, mapping_index_to_node, mapping_entity_to_index, map_index_to_relation = \
        reading_data.create_adjacency_matrix_nt(path_data, literal_representation=literal_map, sparse=True,
                                                relational=True)

    # find the number of nodes
    number_nodes = adjacency_matrix.size()[0]

    # read in the labels, create the masks:
    labels, train_entities, valid_entities, test_entities, label_mapping = \
        reading_data.training_valid_test_set(path_train, path_valid, path_test, mapping_entity_to_index,
                                             adjacency_matrix.size()[0])

    train_mask = torch.tensor([i in train_entities for i in range(len(labels))])
    valid_mask = torch.tensor([i in valid_entities for i in range(len(labels))])
    test_mask = torch.tensor([i in test_entities for i in range(len(labels))])

    # set the indices of the edges and the types of the edges respectively:
    indices_edges = adjacency_matrix.coalesce().indices()[0:2]
    type_edges = adjacency_matrix.coalesce().indices()[2]

    # run in loop
    for weight_decay in [0.001, 0.01, 0.1]:
        for learning_rate in [0.001, 0.005, 0.01, 0.1]:
            for seed in [0, 1, 2]:
                # set the seed as defined, seeds it for pytorch, python and numpy at the same time!
                # none of the previous method use randomization for their processes, so only setting this would suffice
                torch_geometric.seed_everything(seed=seed)

                # create the feature matrix, use vectors of 100 random values for this!
                feature_matrix = torch.rand(size=(number_nodes, 100))

                # create the data object
                data = Data(x=feature_matrix, edge_index=indices_edges, edge_type=type_edges, num_nodes=number_nodes,
                            y=labels.long(), train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)

                # needed for the hyperparameter dictionaries
                nr_relations = data.num_edge_types

                # reduce by 1, as the -1 stands for "no class", therefore it is not needed
                number_classes = len(labels.unique()) - 1

                # add parameter dictionary
                param_dict = None

                # set up the dictionaries used for the parameters, which the GAE does not use
                if model == "GCN":
                    param_dict = {"hidden_nodes": hidden_nodes, "num_classes": number_classes,
                                  "optimizer": optimizer, "learning_rate": learning_rate,
                                  "weight_decay": weight_decay, "nr_epochs": num_epochs}
                elif model == "RGCN":
                    param_dict = {"hidden_nodes": hidden_nodes, "num_classes": number_classes,
                                  "optimizer": optimizer, "learning_rate": learning_rate,
                                  "weight_decay": weight_decay, "nr_epochs": num_epochs, "num_rel": nr_relations}
                elif model == "GAT":
                    param_dict = {"hidden_nodes": hidden_nodes, "num_classes": number_classes,
                                  "optimizer": optimizer, "learning_rate": learning_rate,
                                  "weight_decay": weight_decay, "nr_epochs": num_epochs,
                                  "dropout": dropout_gat, "nr_heads": nr_attention_heads}

                # run the experiment:
                if model != "GAE":
                    result_dict = run_classification_model(data, model, param_dict, seed, literal_map, mapping_index_to_node,
                                                           test=False, record_results=True, path_folder=dataset+"_valid_")
                else:
                    run_gae_model(dataset, data, hidden_nodes, optimizer, learning_rate,
                                  weight_decay, num_epochs, label_mapping, seed, literal_map, mapping_index_to_node)

