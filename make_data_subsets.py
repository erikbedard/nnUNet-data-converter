import os
import glob
from pathlib import Path
import shutil
import nibabel
import numpy as np
import json
import multiprocessing


def make_data_subsets(original_dataset_dir, seg_labels_dict):
    # gather info about dataset
    dataset_name = os.path.basename(original_dataset_dir)
    dataset_num = int(dataset_name.split("Task")[1].split("_")[0])  # Task###_name
    save_dir = Path(original_dataset_dir).parent

    for key in seg_labels_dict.keys():
        if key == 0:  # ignore case (i.e., do not create a background-only dataset)
            continue

        # copy original data set as a starting point for new subset
        dataset_num = dataset_num + 1
        task_num = "%02d" % dataset_num
        new_dataset_name = seg_labels_dict[key].title().replace(" ", "")  # "this label' -> "ThisLabel"
        new_dataset_name = "Task" + task_num + "_" + new_dataset_name
        new_dataset_dir = os.path.join(save_dir, new_dataset_name)
        print("Creating data subset \"" + new_dataset_name + "\"")
        print("  Copying files...")
        shutil.copytree(original_dataset_dir, new_dataset_dir)

        # modify labels as needed
        labelsTr_dir = os.path.join(new_dataset_dir, "labelsTr")
        labelsTs_dir = os.path.join(new_dataset_dir, "labelsTs")
        labelsTr_paths = glob.glob(os.path.join(labelsTr_dir, '*.nii.gz'))
        labelsTs_paths = glob.glob(os.path.join(labelsTs_dir, '*.nii.gz'))
        labels_paths = labelsTr_paths + labelsTs_paths

        # modify label
        print("  Modifying labels...")
        data = list(zip(labels_paths, [key] * len(labels_paths)))
        p = multiprocessing.Pool()
        p.map(modify_labels, data)
        p.close()

        # update json labels
        json_file = os.path.join(new_dataset_dir, "dataset.json")
        with open(json_file, 'r') as file:
            info = json.load(file)
            info["labels"] = {0: "background", 1: seg_labels_dict[key]}
        with open(json_file, 'w') as file:
            json.dump(info, file, indent=4)


def modify_labels(data):
    label_path = data[0]
    class_num = data[1]

    label = nibabel.nifti1.load(label_path)
    label_data = np.array(label.dataobj)

    # set desired class labels to foreground and all other class labels to 0
    label_data[label_data == class_num] = 1
    label_data[label_data != class_num] = 0

    label = nibabel.Nifti1Image(label_data, label.affine, header=label.header)
    nibabel.nifti1.save(label, label_path)

