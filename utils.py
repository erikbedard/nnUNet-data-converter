import nibabel
import dicom2nifti
import numpy as np

import os
import glob
from pathlib import Path
import shutil
import json
import multiprocessing
from typing import Tuple
import random
import math


def get_data_subsets_dirs(original_dataset_dir, seg_labels_dict):
    # gather info about dataset
    dataset_name = os.path.basename(original_dataset_dir)
    dataset_num = int(dataset_name.split("Task")[1].split("_")[0])  # Task###_name
    save_dir = Path(original_dataset_dir).parent

    data_subsets_dirs_dict = {}
    for key in seg_labels_dict.keys():
        if key == 0:  # ignore case (i.e., do not create a background-only dataset)
            continue

        # copy original data set as a starting point for new subset
        dataset_num = dataset_num + 1
        task_num = "%02d" % dataset_num
        new_dataset_name = seg_labels_dict[key].title().replace(" ", "")  # "this label' -> "ThisLabel"
        new_dataset_name = "Task" + task_num + "_" + new_dataset_name
        data_subsets_dirs_dict[key] = os.path.join(save_dir, new_dataset_name)

    return data_subsets_dirs_dict


def make_data_subsets(original_dataset_dir, seg_labels_dict):
    """
    This function takes an existing multi-class segmentation dataset and creates new single-class
    data subsets (one new subset for each foreground class)

    :param original_dataset_dir: where the original dataset is stored
    :param seg_labels_dict: dict of labels
    :return:
    """
    # gather info about dataset
    data_subsets_dirs_dict = get_data_subsets_dirs(original_dataset_dir, seg_labels_dict)

    for key in seg_labels_dict.keys():
        if key == 0:  # ignore case (i.e., do not create a background-only dataset)
            continue

        # copy original data set as a starting point for new subset
        new_dataset_dir = data_subsets_dirs_dict[key]
        print("Creating data subset \"" + os.path.basename(new_dataset_dir) + "\"")
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
        p = multiprocessing.Pool(os.cpu_count()-1)
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

    # set desired class labels to foreground
    label_data = (label_data == class_num).astype('uint8')

    label = nibabel.Nifti1Image(label_data, label.affine, header=label.header)
    nibabel.nifti1.save(label, label_path)


def split_training_data(imagesTr_dir, labelsTr_dir, train_percent=0.7, seed=0):
    """
    This function reads existing training images/labels directories and moves a random percentage of the data into
    new test directories.

    :param imagesTr_dir: directory containing existing training images
    :param labelsTr_dir: directory containing existing training labels
    :param train_percent: percentage of data to retain as training data. The rest will become test data
    :param seed: optional seed for random shuffling of data
    :return:
    """

    # prepare test directories
    parent_dir = Path(imagesTr_dir).parent
    parent_dir2 = Path(labelsTr_dir).parent
    assert parent_dir == parent_dir2

    imagesTs_dir = os.path.join(parent_dir, 'imagesTs')
    labelsTs_dir = os.path.join(parent_dir, 'labelsTs')
    os.makedirs(imagesTs_dir, exist_ok=True)
    os.makedirs(labelsTs_dir, exist_ok=True)

    # get lists of exiting files
    imagesTr_paths = glob.glob(os.path.join(imagesTr_dir, '*.nii.gz'))
    imagesTr_paths.sort()

    labelsTr_paths = glob.glob(os.path.join(labelsTr_dir, '*.nii.gz'))
    labelsTr_paths.sort()

    assert len(imagesTr_paths) == len(labelsTr_paths)
    num_files = len(imagesTr_paths)

    # create new lists for training images/labels
    random.seed(seed)
    random.shuffle(imagesTr_paths)
    random.seed(seed)  # ensure same seed used for both lists
    random.shuffle(labelsTr_paths)

    cutoff = math.floor(train_percent * num_files)
    imagesTs_paths = imagesTr_paths[cutoff:-1]
    labelsTs_paths = labelsTr_paths[cutoff:-1]

    # move files
    for (image, label) in list(zip(imagesTs_paths, labelsTs_paths)):
        shutil.move(image, imagesTs_dir)
        shutil.move(label, labelsTs_dir)


def convert_dicom_to_nifti(input_paths, output_paths):
    data = list(zip(input_paths, output_paths))
    p = multiprocessing.Pool(os.cpu_count()-1)
    p.map(_convert_dicom_to_nifti, data)
    p.close()


def _convert_dicom_to_nifti(data):
    input_dir = data[0]
    new_file_path = data[1]
    print("Converting \"" + input_dir + "\"")
    dicom2nifti.convert_dicom.dicom_series_to_nifti(input_dir, new_file_path, reorient_nifti=False)


#    This code was copied from the nnU-Net repository with some modifications:
#    - removed dependency on batchgenerators
#    - TODO: add dataset stats
#
#
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

def get_identifiers_from_splitted_files(folder: str):
    files = glob.glob(os.path.join(folder, '*.nii.gz'))
    uniques = np.unique([os.path.basename(i)[:-12] for i in files])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")

    with open(os.path.join(output_file), "w") as file:
        json.dump(json_dict, file, indent=4)
