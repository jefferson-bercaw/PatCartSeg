import os


def get_data_path():
    cur_dir = os.getcwd()
    try:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    except KeyError:
        print("No SLURM array task ID found--> Manual task ID employed")
        task_id = 0

    if cur_dir[0:2] == "R:" or cur_dir[0:2] == "C:":  # If on local machine
        if task_id == 5:
            data_path = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data_BMP2"
        elif task_id < 5:
            data_path = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Augmentations/sig" + str(task_id)
    else:  # If on cluster
        if task_id == 5:
            data_path = "/hpc/group/ldefratelab/jrb187/PatCartSeg/Split_Data_BMP2"
        elif task_id < 5:
            data_path = "/hpc/group/ldefratelab/jrb187/PatCartSeg/Augmentations/sig" + str(task_id)

    return data_path