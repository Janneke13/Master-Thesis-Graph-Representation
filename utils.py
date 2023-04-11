import os


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
