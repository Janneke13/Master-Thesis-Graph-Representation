import os


def create_results_folders(folder_name):
    """
    Creates needed folders to store the results in, if they do not exist yet.
    :param folder_name: The additional name of the folder inside the results folder.
    """
    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists("results/" + folder_name):
        os.makedirs("results/" + folder_name)

