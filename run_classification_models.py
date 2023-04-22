import torch
from torch.nn import CrossEntropyLoss
import utils
import os
import csv
from sklearn.metrics import accuracy_score
import GCN_model
import RGCN_model
import GAT_model
import matplotlib.pyplot as plt

"""
File that contains the code to create multiple classification models, as well as run them using a certain dataset.
Results can be printed or stored, and are also returned by the function.
"""


def run_classification_model(data, type_model, model_parameters, seed, test=True, record_results=False,
                             path_folder=None):
    """
    Runs a classification model and prints or stores the results.
    :param data: The PyTorch Geometric data object. Contains the adjacency matrix,labels, and the indices of the sets.
    :param type_model: Defines which classification model is run; can be the GCN, R-GCN or the GAT.
    :param model_parameters: A dictionary containing all the needed model parameters. They differ for the models.
            - hidden_nodes: The number of hidden nodes for the model.
            - num_classes: The number of classes the data can have.
            - optimizer: The optimizer used (option passed in String format, can be Adam, SGD or Adagrad).
            - learning_rate: The learning rate for the model.
            - weight_decay: The amount of weight decay for the model.
            - nr_epochs: The number of epochs the model has to be trained for.
            - nr_heads (GAT-specific): The number of attention heads for the GAT.
            - dropout (GAT-specific): The dropout for the GAT model.
            - num_rel (RGCN-specific): The number of relations the R-GCN needs to take into account.
    :param seed: The random seed used to set it (before entering this function, recording purposes only.
    :param test: Whether the test data is used (if False, the validation data is used).
    :param record_results: If True, the results and the configuration will be recorded in .csv files.
    :param path_folder: Where the results need to be stored (only applicable if record_results is True).
    :return: A dictionary with multiple results, such as the trained model, the loss lists, and the accuracy and
    f1-score.
    """

    # assert whether the chosen model is implemented:
    assert type_model in ["GCN", "RGCN", "GAT"], "type_model must be one of: 'GCN', 'RGCN', 'GAT'."

    # assert whether all the needed parameters are in the parameter dictionary
    assert "hidden_nodes" in model_parameters, "hidden_nodes is not defined."
    assert "num_classes" in model_parameters, "num_classes is not defined."
    assert "optimizer" in model_parameters, "optimizer is not defined."
    assert model_parameters["optimizer"] in ["adam", "sgd", "adagrad"], "optimizer must be one of: " \
                                                                        "'adam', 'sgd', 'adagrad'."
    assert "learning_rate" in model_parameters, "learning_rate is not defined."
    assert "weight_decay" in model_parameters, "weight_decay is not defined"
    assert "nr_epochs" in model_parameters, "nr_epochs is not defined"

    # check whether model-specific parameters are defined.
    if type_model == "GAT":
        assert "nr_heads" in model_parameters, "nr_heads is not defined."
        assert "dropout" in model_parameters, "dropout is not defined."
    elif type_model == "RGCN":
        assert "num_rel" in model_parameters, "num_rel is not defined."

    if record_results:
        # a specialized folder for the specific model and path
        current_test = 1
        while os.path.exists("results/" + type_model + "/" + path_folder + "_" + str(current_test)):
            current_test += 1

        utils.create_results_folders(type_model, path_folder + "_" + str(current_test))

        path = "results/" + type_model + "/" + path_folder + "_" + str(current_test)

        # make a file to record the results
        file_results = open(path + "/results.csv", "w")
        writer_results = csv.writer(file_results)

        # create a header for the classification results
        header = ["Epoch", "CrossEntropyLossTrain", "AccTrain", "CrossEntropyLossTest", "AccTest", "f1Test"]
        writer_results.writerow(header)

        # save the configuration --> so it can be checked later!
        file_config = open(path + "/config.csv", "w")
        writer_config = csv.writer(file_config)

        # create a header and the data for the configuration
        header_config = ["Number_Epochs", "Type_Model", "Hidden_Nodes", "Optimizer", "Learning_Rate", "Weight_Decay",
                         "Test", "Seed"]
        config = [model_parameters["nr_epochs"], type_model, model_parameters["hidden_nodes"],
                  model_parameters["optimizer"], model_parameters["learning_rate"], model_parameters["weight_decay"],
                  test, seed]

        # the GAT and RGCN models have more parameters that need to be stored
        if type_model == "GAT":
            header_config.append("Number_heads")
            config.append(model_parameters["nr_heads"])
            header_config.append("Dropout")
            config.append(model_parameters["dropout"])
        elif type_model == "RGCN":
            header_config.append("Number_relations")
            config.append(model_parameters["num_rel"])

        writer_config.writerow(header_config)
        writer_config.writerow(config)

        # after writing, this file can already be closed, it does not have to be used later on
        file_config.close()

    # ------ CREATE AND RUN THE CLASSIFICATION MODELS --------------

    # create the model based on its type and parameters needed
    model = None

    if type_model == "GCN":
        model = GCN_model.GCN(input_nodes=data.num_node_features, hidden_nodes=model_parameters["hidden_nodes"],
                              output_nodes=model_parameters["num_classes"])
    elif type_model == "RGCN":
        model = RGCN_model.RGCN(input_nodes=data.num_node_features, hidden_nodes=model_parameters["hidden_nodes"],
                                output_nodes=model_parameters["num_classes"], num_rel=model_parameters["num_rel"])
    elif type_model == "GAT":
        model = GAT_model.GAT(input_nodes=data.num_node_features, hidden_nodes=model_parameters["hidden_nodes"],
                              output_nodes=model_parameters["num_classes"], dropout=model_parameters["dropout"],
                              nr_heads=model_parameters["nr_heads"])

    # create the optimizer:
    optim = None

    if model_parameters["optimizer"] == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=model_parameters["learning_rate"],
                                 weight_decay=model_parameters["weight_decay"])
    elif model_parameters["optimizer"] == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=model_parameters["learning_rate"],
                                weight_decay=model_parameters["weight_decay"])
    elif model_parameters["optimizer"] == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=model_parameters["learning_rate"],
                                    weight_decay=model_parameters["weight_decay"])

    # create loss object:
    loss_function = CrossEntropyLoss()

    # create lists to store several attributes for later use:
    loss_list_train = list()
    loss_list_test = list()
    accuracy_list_train = list()
    accuracy_list_test = list()
    f1_list_test = list()

    # check whether cuda can be used and if so, set it as the device!
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # run the training loop:
    for epoch in range(model_parameters["nr_epochs"]):
        # move the dataset to the gpu!
        data.to(device)

        # set the model in train mode and set all the grads at zero
        model.train()
        optim.zero_grad()

        # perform the forward step of the model and calculate the loss
        if type_model == "RGCN":
            output = model(data.x, data.edge_index, data.edge_type)
        else:
            output = model(data.x, data.edge_index)

        loss = loss_function(output[data.train_mask], data.y[data.train_mask])

        # calculate the gradient and perform a step
        loss.backward()
        optim.step()

        # put model to evaluate
        model.eval()

        # put the data to the cpu for this step --> record everything!
        data = data.cpu()

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
            row = [epoch, loss.item(), train_acc, test_loss.item(), test_acc, test_f1]
            writer_results.writerow(row)
        else:
            print("Epoch: ", epoch, ", Training Loss: ", loss.item(), ", Training Acc: ", train_acc,
                  ", Test loss: ", test_loss.item(), ", Test Acc: ", test_acc, ", Test F1: ", test_f1)

    # once out of the loop, as we have to do this once again to get the final results (of the last training step):
    # set everything to evaluation, and do one last step
    data.to(device)  # again, put it in the gpu!

    model.eval()

    if type_model == "RGCN":
        output = model(data.x, data.edge_index, data.edge_type)
    else:
        output = model(data.x, data.edge_index)

    loss = loss_function(output[data.train_mask], data.y[data.train_mask])

    # data to the cpu to compute metrics and record them!
    data = data.cpu()

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
        row = [model_parameters["nr_epochs"], loss.item(), train_acc, test_loss.item(), test_acc, test_f1]
        writer_results.writerow(row)
    else:
        print("Epoch: ", model_parameters["nr_epochs"], ", Training Loss: ", loss.item(), ", Training Acc: ", train_acc,
              ", Test loss: ", test_loss.item(), ", Test Acc: ", test_acc, ", Test F1: ", test_f1)

    # make a dictionary of results and return this:
    result_dict = {"model": model, "loss_list_train": loss_list_train, "loss_list_test": loss_list_test,
                   "accuracy_train": accuracy_list_train, "accuracy_test": accuracy_list_test, "f1_test": f1_list_test}

    # make plots and save them!

    # loss
    plt.plot(loss_list_train, label="Training")
    if test:
        plt.plot(loss_list_test, label="Test")
    else:
        plt.plot(loss_list_test, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig(path + "/loss_plot.jpeg", dpi=300)

    # accuracy
    plt.figure()
    plt.plot(accuracy_list_train, label="Training")
    if test:
        plt.plot(accuracy_list_test, label="Test")
    else:
        plt.plot(accuracy_list_test, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(path + "/acc_plot.jpeg", dpi=300)

    return result_dict
