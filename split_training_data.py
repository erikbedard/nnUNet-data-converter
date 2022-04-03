import glob
import os
import random
import math
import shutil
from pathlib import Path


def split_training_data(imagesTr_dir, labelsTr_dir, train_percent=0.7, seed=0):
    # prepare test directories
    parent_dir = Path(imagesTr_dir).parent
    parent_dir2 = Path(labelsTr_dir).parent
    assert parent_dir == parent_dir2

    imagesTs_dir = os.path.join(parent_dir, 'imagesTs')
    labelsTs_dir = os.path.join(parent_dir, 'labelsTs')
    os.mkdir(imagesTs_dir)
    os.mkdir(labelsTs_dir)

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
