import os


def get_data_path(dataset):
    cur_dir = os.getcwd()
    sep = os.sep
    if "Paranjape" not in dataset and "Crook" not in dataset and "Owusu-Akyaw" not in dataset:
        if cur_dir[0:2] == "R:" or cur_dir[0:2] == "C:" or "DefratePrivate" in cur_dir:  # If on local machine
            data_path = "R:" + sep + "DefratePrivate" + sep + "Bercaw" + sep + "Patella_Autoseg" + sep + f"Split_Data_{dataset}"
        else:  # If on cluster
            data_path = sep + "hpc" + sep + "group" + sep + "ldefratelab" + sep + "jrb187" + sep + "PatCartSeg" + sep + f"Split_Data_{dataset}"
    elif "Paranjape" in dataset:
        if cur_dir[0:2] == "R:" or cur_dir[0:2] == "C:" or "DefratePrivate" in cur_dir:  # If on local machine
            data_path = "R:" + sep + "DefratePrivate" + sep + "Bercaw" + sep + "Patella_Autoseg" + sep + dataset
        else:  # If on cluster
            data_path = sep + "hpc" + sep + "group" + sep + "ldefratelab" + sep + "jrb187" + sep + "PatCartSeg" + sep + dataset
    elif "Crook" in dataset:
        if cur_dir[0:2] == "R:" or cur_dir[0:2] == "C:" or "DefratePrivate" in cur_dir:  # If on local machine
            data_path = "R:" + sep + "DefratePrivate" + sep + "Bercaw" + sep + "Patella_Autoseg" + sep + dataset
        else:  # If on cluster
            data_path = sep + "hpc" + sep + "group" + sep + "ldefratelab" + sep + "jrb187" + sep + "PatCartSeg" + sep + dataset
    elif "Owusu-Akyaw" in dataset:
        if cur_dir[0:2] == "R:" or cur_dir[0:2] == "C:" or "DefratePrivate" in cur_dir:  # If on local machine
            data_path = "R:" + sep + "DefratePrivate" + sep + "Bercaw" + sep + "Patella_Autoseg" + sep + dataset
        else:  # If on cluster
            data_path = sep + "hpc" + sep + "group" + sep + "ldefratelab" + sep + "jrb187" + sep + "PatCartSeg" + sep + dataset
    return data_path
