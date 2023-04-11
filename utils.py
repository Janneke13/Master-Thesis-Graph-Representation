import os
from sklearn.metrics import accuracy_score, f1_score


def create_results_folders(folder_name, folder_in_folder=None):
    """
    Creates needed folders to store the results in, if they do not exist yet.
    :param folder_name: The additional name of the folder inside the 'results' folder.
    :param folder_in_folder: If an additional folder needs to be made within the folder, create this as well.
    """
    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists("results/" + folder_name):
        os.makedirs("results/" + folder_name)

    if folder_in_folder:
        if not os.path.exists("results/" + folder_name + "/" + folder_in_folder):
            os.makedirs("results/" + folder_name + "/" + folder_in_folder)


def calculate_class_metrics(output, labels, predictions, loss_function, mask):
    """
    Calculates several metrics for classification.
    :param output: The output of the epoch (the embeddings).
    :param labels: The true labels.
    :param predictions: The predictions of the current epoch.
    :param loss_function: The loss function used for the model.
    :param mask: The mask to calculate the metrics over (can be, for example, the train, validation or test mask.)
    :return: The calculated loss, accuracy, and f1 score.
    """
    loss = loss_function(output[mask], labels[mask])
    acc = accuracy_score(labels[mask], predictions[mask])
    f1 = f1_score(labels[mask], predictions[mask], average="weighted")
    return loss, acc, f1
