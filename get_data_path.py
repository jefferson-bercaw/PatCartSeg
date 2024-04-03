import os


def get_data_path():
    cur_dir = os.getcwd()
    if cur_dir[0:2] == "R:" | cur_dir[0:2] == "C:":  # If on local machine
        data_path = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data_BMP2"
    else:  # If on cluster
        data_path = "/hpc/group/ldefratelab/jrb187/PatCartSeg/Split_Data_BMP2"

    return data_path