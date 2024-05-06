import os

def get_data_path(dataset):
    cur_dir = os.getcwd()

    if cur_dir[0:2] == "R:" or cur_dir[0:2] == "C:":  # If on local machine
        data_path = f"R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data_{dataset}"
    else:  # If on cluster
        data_path = f"/hpc/group/ldefratelab/jrb187/PatCartSeg/Split_Data_{dataset}"
    return data_path