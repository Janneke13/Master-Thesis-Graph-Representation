import torch
from torch_geometric.nn import GCNConv
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import utils
import os
import csv
from sklearn.metrics import accuracy_score, f1_score


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


def run_gcn_model(data, hidden_nodes, num_classes, optimizer, learning_rate, weight_decay,
                  nr_epochs, test=True, record_results=False, path_folder=None):
    """
    Runs the GCN model and stores the results.
    :param data: The PyTorch Geometric data object. Contains the adjacency matrix,labels, and the indices of the sets.
    :param hidden_nodes: The number of hidden nodes for the GCN model.
    :param num_classes: The number of classes the data can have.
    :param optimizer: The optimizer used (option passed in String format).
    :param learning_rate: The learning rate for the GCN model.
    :param weight_decay: The amount of weight decay for the GCN model.
    :param nr_epochs: The number of epochs the model has to be trained for.
    :param test: Whether the test data is used (if False, the validation data is used).
    :param record_results: If True, the results and the configuration will be recorded in .csv files.
    :param path_folder: Where the results need to be stored (only applicable if record_results is True).
    :return: A dictionary with multiple results, such as the trained model, the loss lists, and the accuracy and
    f1-score.
    """

    # assert whether aspects of the input data is properly defined
    assert optimizer in ["adam", "sgd", "adagrad"], "Optimizer must be one of: 'adam', 'sgd', 'adagrad'."
    # add more asserts if needed

    if record_results:
        # a specialized folder for the specific GCN run type
        utils.create_results_folders("GCN", path_folder)

        # find out which filename is still available
        current_test = 1
        while os.path.exists("results/GCN/" + path_folder + "/" + str(current_test) + ".csv"):
            current_test += 1

        # make a file to record the results
        file_results = open("results/GCN/" + path_folder + "/" + str(current_test) + ".csv", "w")
        writer_results = csv.writer(file_results)

        # create a header
        header = ["Epoch", "CrossEntropyLossTrain", "AccTrain", "CrossEntropyLossTest", "AccTest", "f1Test"]
        writer_results.writerow(header)

        # save the configuration --> so it can be checked later!
        file_config = open("results/GCN/" + path_folder + "/" + str(current_test) + "_config.csv", "w")
        writer_config = csv.writer(file_config)

        # create a header and the data for the configuration, write this in
        header_config = ["Number_Epochs", "Hidden_Nodes", "Optimizer", "Learning_Rate", "Weight_Decay", "Test"]
        config = [nr_epochs, hidden_nodes, optimizer, learning_rate, weight_decay, test]
        writer_config.writerow(header_config)
        writer_config.writerow(config)
        file_config.close()  # this file can already be closed, it does not have to be used later on

    # ------ CREATE AND RUN GCN MODEL --------------
    # create the GCN model
    model = GCN(data.num_node_features, hidden_nodes, num_classes)

    # create the optimizer:
    if optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # create loss object:
    loss_function = CrossEntropyLoss()

    # create lists to store several attributes for later use:
    loss_list_train = list()
    loss_list_test = list()
    accuracy_list_train = list()
    accuracy_list_test = list()
    f1_list_test = list()

    # run the training loop:
    for epoch in range(nr_epochs):

        # set the model in train mode and set all the grads at zero
        model.train()
        optim.zero_grad()

        # perform the forward step of the model and calculate the loss
        output = model(data.x, data.edge_index)
        loss = loss_function(output[data.train_mask], data.y[data.train_mask])

        # calculate the gradient and perform a step
        loss.backward()
        optim.step()

        # put model to evaluate
        model.eval()

        # get the predictions
        predictions = output.argmax(dim=1)
        train_acc = accuracy_score(data.y[data.train_mask], predictions[data.train_mask])

        # now, calculate, store and record everything:
        if test:
            test_loss, test_acc, test_f1 = utils.calculate_class_metrics(output, data.y,
                                                                         predictions, loss_function, data.test_mask)
        else:
            test_loss, test_acc, test_f1 = utils.calculate_class_metrics(output, data.y,
                                                                         predictions, loss_function, data.val_mask)

        # add everything to the lists
        loss_list_train.append(loss.item())
        loss_list_test.append(test_loss.item())
        accuracy_list_train.append(train_acc)
        accuracy_list_test.append(test_acc)
        f1_list_test.append(test_f1)

        # record results, or print them:
        if record_results:
            row = [epoch, loss.item(), train_acc, test_loss.item(),test_acc, test_f1]
            writer_results.writerow(row)
        else:
            print("Epoch: " , epoch, ", Training Loss: ", loss.item(), ", Training Acc: ",  train_acc,
                  ", Test loss: ", test_loss.item(), ", Test Acc: ", test_acc, ", Test F1: ", test_f1)

    # out of the loop, as we have tp do this once again to get the final results (of the last training step:
    # set everything to evaluation, and do one last step
    model.eval()
    output = model(data.x, data.edge_index)
    loss = loss_function(output[data.train_mask], data.y[data.train_mask])

    # get the predictions and the training accuracy
    predictions = output.argmax(dim=1)
    train_acc = accuracy_score(data.y[data.train_mask], predictions[data.train_mask])

    if test:
        test_loss, test_acc, test_f1 = utils.calculate_class_metrics(output, data.y,
                                                                     predictions, loss_function, data.test_mask)
    else:
        test_loss, test_acc, test_f1 = utils.calculate_class_metrics(output, data.y,
                                                                     predictions, loss_function, data.val_mask)

    # again, record the results if needed
    if record_results:
        row = [nr_epochs, loss.item(), train_acc, test_loss.item(), test_acc, test_f1]
        writer_results.writerow(row)
    else:
        print("Epoch: ", nr_epochs, ", Training Loss: ", loss.item(), ", Training Acc: ", train_acc,
              ", Test loss: ", test_loss.item(), ", Test Acc: ", test_acc, ", Test F1: ", test_f1)

    # make a dictionary of results and return this:
    result_dict = {"model": model, "loss_list_train": loss_list_train, "loss_list_test":loss_list_test,
                   "accuracy_train": accuracy_list_train, "accuracy_test": accuracy_list_test, "f1_test": f1_list_test}

    return result_dict

