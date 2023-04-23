from run_classification_models import run_classification_model
import torch
from torch_geometric.data import Data
import reading_data
import torch_geometric
from run_graph_auto_encoder import run_gae_model


"""
Runs experiments in loops --> only reads in data once.
Used for hyperparameter optimization for convenience, as it speeds up the process a lot.

For this, the parameters and loops need to be created manually.
It changes one hyperparameter while keeping the other ones constant.
"""


dataset = "aifb"
# seed will be ran in loops --> 1, 2, 3
literal_map = "separate"
model = "GCN"
hidden_nodes = 16
# learning_rate = 0.001 -- testing the learning rate
optimizer ="adam"
num_epochs = 250
weight_decay = 0
dropout_gat = 0.6
nr_attention_heads = 8

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

# read in the data, set relational to True, as it is not taken into account for non-relational data, so it does no harm
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
for learning_rate in [0.001, 0.005, 0.01, 0.1]:
    for seed in [0, 1, 2]:
        # set the seed as defined, seeds it for pytorch, python and numpy at the same time!
        # none of the previous method use randomization for their processes, so only setting this now would suffice
        torch_geometric.seed_everything(seed=seed)

        # create the feature matrix, use vectors of 100 random values for this!
        feature_matrix = torch.rand(size=(number_nodes, 100))

        # create the data object
        data = Data(x=feature_matrix, edge_index=indices_edges, edge_type=type_edges, num_nodes=number_nodes,
                    y=labels.long(), train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)

        # needed for the hyperparameter dictionaries
        nr_relations = data.num_edge_types
        number_classes = len(labels.unique()) - 1  # reduce by 1, as the 0 stands for "no class", therefore it is not needed

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
            result_dict = run_classification_model(data, model, param_dict, seed, literal_map,
                                                   test=False, record_results=True, path_folder=dataset+"_valid_" )
        else:
            run_gae_model(dataset, data, hidden_nodes, optimizer, learning_rate,
                          weight_decay, num_epochs, label_mapping, seed, literal_map)

