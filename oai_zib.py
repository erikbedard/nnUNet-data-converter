"""This module is used to extract/standardize the knee MRI data from the OAI data set and the segmentation labels
from the OAI-ZIB dataset into 'images' and 'labels' directories.
It is assumed that:
    (1) the OAI baseline data for 4796 subjects has been downloaded from https://nda.nih.gov/oai/ and the zip file
        been extracted (e.g. to a directory such as '~/Downloads/Package_1198790/results/00m')
    (2) the OAI-ZIB data of 507 manual segmentations has been downloaded from https://pubdata.zib.de/ and the zip file
        has been extract (e.g. to a directory such as '~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations')


Usage:
python oai_zib.py oai_dir oai_zib_dir save_dir

Example:
python oai_zib.py
    "~/Downloads/Package_1198790/results/00m"
    "~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations"
    "~/data/oai-zib"

"""

import SimpleITK as sitk
import numpy as np
import cv2

import sys
import argparse
import os
import glob
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract OAI-ZIB images and labels.")
    parser.add_argument(dest="oai_dir", type=str,
                        help="Directory where the OIA data is stored, e.g. \"~/Downloads/Package_1198790/results/00m\"")
    parser.add_argument(dest="oai_zib_dir", type=str,
                        help="Directory where OAI-ZIB data is stored, e.g. \"~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations\"")
    parser.add_argument(dest="save_dir", type=str,
                        help="Directory where the images and labels will be saved, e.g. \"~/data/oai-zib\"")

    args = parser.parse_args()

    # validate oai_data_dir
    dir0C2 = os.path.join(args.oai_dir, "0.C.2")
    dir0E1 = os.path.join(args.oai_dir, "0.E.1")
    if not (os.path.exists(dir0C2) and os.path.exists(dir0E1)):
        print("The following directories must exist:")
        print(dir0C2)
        print(dir0E1)
        sys.exit("ERROR: Invalid OAI data directory was specified.")

    # validate oai_zib_data_dir
    oai_mri_paths = os.path.join(args.oai_zib_dir, "segmentation", "doc", "oai_mri_paths.txt")
    oai_zib_masks_dir = os.path.join(args.oai_zib_dir, "segmentation", "segmentation_masks")
    if not (os.path.exists(oai_mri_paths) and os.path.exists(oai_zib_masks_dir)):
        print("The following file and directory must exist:")
        print(oai_mri_paths)
        print(oai_zib_masks_dir)
        sys.exit("ERROR: Invalid OAI-ZIB data directory was specified.")

    # extract the relevant MRIs
    images_dir = os.path.join(args.save_dir, "images")
    copy_mri_subset(args.oai_dir, oai_mri_paths, images_dir)

    # convert the masks to png files
    labels_dir = os.path.join(args.save_dir, "labels")
    convert_all_masks(oai_zib_masks_dir, labels_dir)


def load_itk_image(filename):
    itk_image = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itk_image)
    return numpy_image


def convert_mhd_to_png(mhd_filepath, save_name=None, save_dir=None):
    """
    Convert an .mhd (and its .raw file pair) to a series of sequentially-numbered .png files
    :param mhd_filepath: full path to the .mhd file to be converted
    :param save_name: by default, the .png files will have the same name as the .mhd file. 'save_name' can optionally be
        set to user-specified name.
    :param save_dir: by default, the .png files will be saved in the same location as the .mhd file. 'save_dir' can
        optionally be set to a user-specified location.

    The .png files will be saved with the following structure, where 'XXXX' denotes the index of the last image in the
    sequence with leading zeros:

    save_dir/save_name
    ├── save_name-0000.png
    ├── save_name-0001.png
    ...
    └──save_name-XXXX.png
    """

    # create save directory
    if save_name is None:
        # use name of mhd file as save_name
        save_name = Path(mhd_filepath).stem
    if save_dir is None:
        # save png files in same location as input
        save_dir = os.path.dirname(mhd_filepath)
    save_subdir = os.path.join(save_dir, save_name)
    os.makedirs(save_subdir, exist_ok=True)

    # process image
    vol = load_itk_image(mhd_filepath)
    num_slices = np.size(vol, axis=2)
    for i in range(num_slices):
        im = vol[:, :, i]
        idx = format(i, '04')
        file_name = save_name + '-' + str(idx) + '.png'
        im_path = os.path.join(save_subdir, file_name)
        cv2.imwrite(im_path, im)


def convert_all_masks(oai_zib_masks_dir, save_dir):
    """
    Convert all the segmentation masks from the OAI ZIB data set to .png sequences
    :param oai_zib_masks_dir: path of the OAI ZIB 'segmentation_masks' folder
    :param save_dir: path of the directory where the .png sequences will be saved
    """
    assert os.path.basename(oai_zib_masks_dir) == "segmentation_masks"

    mhd_files = glob.glob(os.path.join(oai_zib_masks_dir, '*.mhd'))
    mhd_files.sort()
    num_files = len(mhd_files)

    print(str(num_files) + " \'.mhd\' files found in " + oai_zib_masks_dir)
    for i in range(num_files):
        this_file = mhd_files[i]
        file_name = os.path.basename(this_file)
        save_name = file_name.split('.')[0]  # only keep first part of file name as the save name
        print("[" + str(i+1) + " of " + str(num_files) + "] converting \"" + file_name + "\"")
        convert_mhd_to_png(this_file, save_name=save_name, save_dir=save_dir)


def copy_mri_subset(oai_data_dir, oai_mri_paths, save_dir):
    """
    Copy the relevant MRIs for which there exists a segmentation mask (as listed by the ZIB dataset). The MRI files will
    also be renamed to include the MRI ID as a prefix and to add a '.dcm' extension
    :param oai_data_dir: path of the directory where the baseline OAI data is stored (i.e. /Package_1198790/results/00m)
    :param oai_mri_paths: file path of the 'oai_mri_paths.txt' file from the OAI ZIB dataset
    :param save_dir: directory where the MRIs will be copied

    The copied files will be saved with the following structure, where 'XXXX' denotes the index of the last image in the
    sequence with leading zeros. The 'mri_id' is taken from the 'oai_mri_paths' file.

    save_dir/mri_id
    ├── mri_id-0000.png
    ├── mri_id-0001.png
    ...
    └──mri_id-XXXX.png
    """

    path_dict = get_mri_list(oai_mri_paths)

    # loop through all scans which are referenced
    num_scans = len(path_dict)
    print(str(num_scans) + " MRI scans are referenced in \"" + oai_mri_paths + "\"")
    i = 1
    for key in path_dict.keys():
        value = path_dict[key]
        old_dir = os.path.join(oai_data_dir, value)
        mri_files = glob.glob(os.path.join(old_dir, '*'))
        mri_files.sort()
        num_slices = len(mri_files)

        # copy all MRI slices to new destination
        print("[" + str(i) + " of " + str(num_scans) + "] copying MRI \"" + key + "\"")
        new_dir = os.path.join(save_dir, key)
        os.makedirs(new_dir, exist_ok=True)
        for j in range(num_slices):
            old_file_path = mri_files[j]
            save_name = key
            idx = format(j, '04')
            new_file_name = save_name + '-' + str(idx) + '.dcm'
            new_file_path = os.path.join(new_dir, new_file_name)
            shutil.copy2(old_file_path, new_file_path)

        i += 1


def get_mri_list(oai_mri_paths):
    """
    Create a dictionary from the contents of the 'oai_mri_paths' file
    :param oai_mri_paths: file path of the 'oai_mri_paths.txt' file from the OAI ZIB dataset
    :return path_dict: dictionary of the MRI names and their relative file paths
    """
    assert os.path.basename(oai_mri_paths) == "oai_mri_paths.txt"

    path_dict = {}
    with open(oai_mri_paths, 'r') as data:
        for line in data.readlines():
            (name, path) = line.split(':')
            name = name.strip()
            path = path.strip()

            # reconstruct path for this os
            path = path.split('/')
            path = os.path.join(*path)

            path_dict[name] = path

    return path_dict


if __name__ == '__main__':
    main()
