import argparse
from run_classification_models import run_classification_model
import torch
from torch_geometric.data import Data
import reading_data
import torch_geometric
from run_graph_auto_encoder import run_gae_model
import os
import json

"""
This file is used to run the model and save the outputs. Command line can be used to put in the different arguments.

It runs the model and saves the output of it in a folder. 
"""

# use a parser so the command line can be used to set up everything (useful when used with DAS-5)
parser = argparse.ArgumentParser()

# basic arguments - such as the dataset and the random seed added
parser.add_argument("--dataset", type=str, default='aifb', help="The used dataset, can be AIFB+, MUTAG or DMG777k.")
parser.add_argument("--test", type=bool, default=False, help="Needs to be set to True when test data is used.")
parser.add_argument("--literal_map", type=str, default="filtered", help="The mapping technique used for the literals.")

# arguments for the model itself
parser.add_argument("--model", type=str, default="GCN", help="The model used.")
parser.add_argument("--hidden_nodes", type=int, default=16, help="The amount of hidden nodes in the GNN.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate used to train the GNN.")
parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer used to train the GNN.")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs used for training.")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay used during training.")

# arguments specifically for the GAT (ignored when GAT is not used)
parser.add_argument("--dropout_gat", type=float, default=0.6, help="Dropout probability used for GAT.")
parser.add_argument("--nr_attention_heads", type=int, default=8, help="Number of attention heads used by the GAT.")

# read all the arguments from the command line
arguments = parser.parse_args()

# some asserts, to make none of the modes defined are nonexistent
assert arguments.dataset in ['aifb', 'mutag', 'dmg777k', 'synth'], "Dataset needs to be one of the following: 'aifb', 'mutag'," \
                                                          " 'dmg777k, 'synth'"
assert arguments.model in ["GAE", "GCN", "RGCN", "GAT"], "Model needs to be one of the following: 'GAE', 'GCN', " \
                                                         "'RGCN', 'GAT'."
assert arguments.optimizer in ['sgd', 'adam', 'adagrad'], "Optimizer needs to be one of the following: 'sgd', 'adam, " \
                                                          " 'adagrad."
assert arguments.literal_map in ['filtered', 'collapsed', 'separate', 'all-to-one'], "Literal mapping needs to be one" \
                                                                                     "of the following: 'filtered', " \
                                                                                     "'collapsed', 'separate', " \
                                                                                     "'all-to-one'."

# read in the data and create data objects:

# set the correct path:
if arguments.dataset == 'aifb':
    path_data = "data/aifb/gz_files/aifb+.nt.gz"
    path_train = "data/aifb/gz_files/aifb+_train_set.nt.gz"
    path_valid = "data/aifb/gz_files/aifb+_valid_set.nt.gz"
    path_test = "data/aifb/gz_files/aifb+_test_set.nt.gz"
elif arguments.dataset == 'mutag':
    path_data = "data/mutag/gz_files/mutag.nt.gz"
    path_train = "data/mutag/gz_files/mutag_train_set.nt.gz"
    path_valid = "data/mutag/gz_files/mutag_valid_set.nt.gz"
    path_test = "data/mutag/gz_files/mutag_test_set.nt.gz"
elif arguments.dataset == "synth":
    path_data = "data/synth/context.nt.gz"
    path_train = "data/synth/train.nt.gz"
    path_valid = "data/synth/valid.nt.gz"
    path_test = "data/synth/test.nt.gz"
else:  # dmg777k
    path_data = "data/dmg777k/gz_files/dmg777k_stripped.nt.gz"
    path_train = "data/dmg777k/gz_files/dmg777k_train_set.nt.gz"
    path_valid = "data/dmg777k/gz_files/dmg777k_valid_set.nt.gz"
    path_test = "data/dmg777k/gz_files/dmg777k_test_set.nt.gz"

# this is all deterministic so seed does not matter!
# read in the data, set relational to True, as it is not taken into account for non-relational data, so it does no harm
adjacency_matrix, mapping_index_to_node, mapping_entity_to_index, map_index_to_relation = \
    reading_data.create_adjacency_matrix_nt(path_data, literal_representation=arguments.literal_map, sparse=True,
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

# save for later -- in case this is needed
if not os.path.exists('results/' + arguments.model + "/test_" + arguments.dataset + "_" + arguments.literal_map):
    os.makedirs('results/' + arguments.model + "/test_" + arguments.dataset + "_" + arguments.literal_map)

with open('results/' + arguments.model + "/test_" + arguments.dataset + "_" + arguments.literal_map + '_final/mapping_ent_to_ind.json', 'w') as file:
    json.dump(mapping_entity_to_index, file)

with open('results/' + arguments.model + "/test_" + arguments.dataset + "_" + arguments.literal_map + '_final/mapping_ind_to_rel.json', 'w') as file:
    json.dump(map_index_to_relation, file)

with open('results/' + arguments.model + "/test_" + arguments.dataset + "_" + arguments.literal_map + '_final/label_mapping.json', 'w') as file:
    json.dump(label_mapping, file)

# run it 10 times with different seeds
for seed in range(1, 11):
    # set the seed as defined, seeds it for pytorch, python and numpy at the same time!
    torch_geometric.seed_everything(seed=seed)

    # create the feature matrix, use vectors of 100 random values for this!
    feature_matrix = torch.rand(size=(number_nodes, 100))

    # set the indices of the edges and the types of the edges respectively:
    indices_edges = adjacency_matrix.coalesce().indices()[0:2]
    type_edges = adjacency_matrix.coalesce().indices()[2]

    # create the data object
    data = Data(x=feature_matrix, edge_index=indices_edges, edge_type=type_edges, num_nodes=number_nodes,
                y=labels.long(), train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)

    # needed for the hyperparameter dictionaries
    nr_relations = data.num_edge_types
    number_classes = len(labels.unique()) - 1  # reduce by 1, as the 0 stands for "no class", therefore it is not needed

    # add parameter dictionary
    param_dict = None

    # set up the dictionaries used for the parameters, which the GAE does not use
    if arguments.model == "GCN":
        param_dict = {"hidden_nodes": arguments.hidden_nodes, "num_classes": number_classes,
                      "optimizer": arguments.optimizer, "learning_rate": arguments.learning_rate,
                      "weight_decay": arguments.weight_decay, "nr_epochs": arguments.num_epochs}
    elif arguments.model == "RGCN":
        param_dict = {"hidden_nodes": arguments.hidden_nodes, "num_classes": number_classes,
                      "optimizer": arguments.optimizer, "learning_rate": arguments.learning_rate,
                      "weight_decay": arguments.weight_decay, "nr_epochs": arguments.num_epochs,
                      "num_rel": nr_relations}
    elif arguments.model == "GAT":
        param_dict = {"hidden_nodes": arguments.hidden_nodes, "num_classes": number_classes,
                      "optimizer": arguments.optimizer, "learning_rate": arguments.learning_rate,
                      "weight_decay": arguments.weight_decay, "nr_epochs": arguments.num_epochs,
                      "dropout": arguments.dropout_gat, "nr_heads": arguments.nr_attention_heads}

    # run the experiment:
    if arguments.model != "GAE":
        result_dict = run_classification_model(data, arguments.model, param_dict, seed, arguments.literal_map, mapping_index_to_node,
                                               test=arguments.test, record_results=True, path_folder=arguments.dataset)
    else:
        run_gae_model(arguments.dataset, data, arguments.hidden_nodes, arguments.optimizer, arguments.learning_rate,
                      arguments.weight_decay, arguments.num_epochs, label_mapping, seed, arguments.literal_map,
                      mapping_index_to_node)
