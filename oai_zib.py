import dicom2nifti
import SimpleITK as sitk
import multiprocessing
import nibabel

import sys
import os
import argparse
import glob

from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles
from batchgenerators.utilities.file_and_folder_operations import save_json


def main():
    print("\n")
    print("##############################")
    print("##  OIA-ZIB Data Converter  ##")
    print("##############################")
    print("")
    parser = argparse.ArgumentParser(description="Extract OAI-ZIB images and labels.")
    parser.add_argument(dest="oai_dir", type=str,
                        help="Directory where the OIA data is stored, e.g. \"~/Downloads/Package_1198790/results/00m\"")
    parser.add_argument(dest="oai_zib_dir", type=str,
                        help="Directory where OAI-ZIB data is stored, e.g. \"~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations\"")
    parser.add_argument(dest="save_dir", type=str,
                        help="Directory where the images and labels will be saved, e.g. \"~/data/oai-zib\"")

    args = parser.parse_args()

    # validate oai_dir
    dir0C2 = os.path.join(args.oai_dir, "0.C.2")
    dir0E1 = os.path.join(args.oai_dir, "0.E.1")
    if not (os.path.exists(dir0C2) and os.path.exists(dir0E1)):
        print("The following directories must exist:")
        print(dir0C2)
        print(dir0E1)
        sys.exit("ERROR: Invalid OAI data directory was specified.")

    # validate oai_zib_dir
    oai_mri_paths = os.path.join(args.oai_zib_dir, "segmentation", "doc", "oai_mri_paths.txt")
    oai_zib_masks_dir = os.path.join(args.oai_zib_dir, "segmentation", "segmentation_masks")
    if not (os.path.exists(oai_mri_paths) and os.path.exists(oai_zib_masks_dir)):
        print("The following file and directory must exist:")
        print(oai_mri_paths)
        print(oai_zib_masks_dir)
        sys.exit("ERROR: Invalid OAI-ZIB data directory was specified.")

    # convert MRIs
    imagesTr_dir = os.path.join(args.save_dir, "imagesTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    convert_mri_subset(args.oai_dir, oai_mri_paths, imagesTr_dir)

    # convert masks
    labelsTr_dir = os.path.join(args.save_dir, "labelsTr")
    os.makedirs(labelsTr_dir, exist_ok=True)
    convert_all_masks(oai_zib_masks_dir, labelsTr_dir)

    align_images_and_labels(imagesTr_dir, labelsTr_dir)

    imagesTs_dir = None

    # prepare dataset JSON file
    output_file = os.path.join(args.save_dir, "dataset.json")
    print("\nCreating \"" + output_file + "\"")
    modalities = ["MRI"]
    labels = {0: "background",
              1: "femoral bone",
              2: "femoral cartilage",
              3: "tibial bone",
              4: "tibial cartilage"}
    dataset_name = "OAI-ZIB Knee"
    license = ""
    dataset_description = "507 DESS MRIs from the OAI database with manual segmentations provided by ZIB"
    dataset_reference = "https://nda.nih.gov/oai/ and https://pubdata.zib.de/"
    dataset_release = "0.0"

    generate_dataset_json(output_file, imagesTr_dir, imagesTs_dir, modalities,
                          labels, dataset_name, license, dataset_description,
                          dataset_reference, dataset_release)

    print("Finished!")


def convert_all_masks(oai_zib_masks_dir, save_dir):
    """
    Convert all the segmentation masks from the OAI ZIB data set to the nifti format with the naming convention required
    by nnU-Net.

    :param oai_zib_masks_dir: path of the OAI ZIB 'segmentation_masks' folder
    :param save_dir: path of the directory where the nifti files will be saved
    """

    assert os.path.basename(oai_zib_masks_dir) == "segmentation_masks"

    mhd_files = glob.glob(os.path.join(oai_zib_masks_dir, '*.mhd'))
    num_files = len(mhd_files)

    # create data list for parallel processing
    t0 = mhd_files
    t1 = [save_dir] * num_files
    data = list(zip(t0, t1))

    print("\n" + str(num_files) + " \'.mhd\' files found in " + oai_zib_masks_dir)

    # process masks
    p = multiprocessing.Pool()
    p.map(convert_mhd_to_nifti, data)


def convert_mri_subset(oai_data_dir, oai_mri_paths, save_dir):
    """
    Convert the MRIs into nifti files for which there exists a segmentation mask (as listed by the ZIB dataset). The
    naming convention required by nnU-Net is also followed.

    :param oai_data_dir: path of the directory where the baseline OAI data is stored (i.e. /Package_1198790/results/00m)
    :param oai_mri_paths: file path of the 'oai_mri_paths.txt' file from the OAI ZIB dataset
    :param save_dir: directory where the nifti files will be saved
    """

    path_dict = get_mri_list(oai_mri_paths)
    num_scans = len(path_dict)

    # create data list for parallel processing
    t0 = [oai_data_dir] * num_scans
    t1 = path_dict.values()
    t2 = [save_dir] * num_scans
    t3 = path_dict.keys()
    data = list(zip(t0, t1, t2, t3))

    print(str(num_scans) + " MRI scans in:\n  \"" + oai_data_dir + "\"\nwill be converted and saved to:\n  \"" + save_dir + "\"\n")
    input("Press Enter to continue...")

    # process masks
    p = multiprocessing.Pool()
    p.map(convert_dicom_to_nifti, data)


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


def convert_dicom_to_nifti(data):
    # create read image filepath
    data_dir = data[0]
    image_rel_path = data[1]
    old_dir = os.path.join(data_dir, image_rel_path)
    print("Converting \"" + image_rel_path + "\"")

    # create write image filepath
    out_dir = data[2]
    image_id = data[3]
    new_file_name = 'knee_' + image_id + '_0000.nii.gz'
    new_file_path = os.path.join(out_dir, new_file_name)
    dicom2nifti.convert_dicom.dicom_series_to_nifti(old_dir, new_file_path, reorient_nifti=False)


def convert_mhd_to_nifti(data):
    image_path = data[0]
    itk_image = sitk.ReadImage(image_path)

    file_name = os.path.basename(image_path)
    image_id = file_name.split('.')[0]  # only keep first part of file name as the image ID

    print("Converting \"" + file_name + "\"")
    save_dir = data[1]
    new_file_name = 'knee_' + image_id + '.nii.gz'
    new_file_path = os.path.join(save_dir, new_file_name)

    sitk.WriteImage(itk_image, new_file_path)


def align_images_and_labels(images_dir, labels_dir):
    image_paths = glob.glob(os.path.join(images_dir, '*.nii.gz'))
    image_paths.sort()

    label_paths = glob.glob(os.path.join(labels_dir, '*.nii.gz'))
    label_paths.sort()

    # create data list for parallel processing
    data = zip(image_paths, label_paths)

    print("\nRe-aligning segmentation labels...")

    # process masks
    p = multiprocessing.Pool()
    p.map(realign, data)


def realign(data):
    im_path = data[0]
    lbl_path = data[1]

    print("Re-aligning: " + lbl_path)

    image = nibabel.nifti1.load(im_path)
    label = nibabel.nifti1.load(lbl_path)


    # swap label axes so that they are consistent with image axes
    # label_data = label_data.astype(np.ushort) # images are ushort, so make labels the same
    label_data = np.array(label.dataobj)
    label_data = np.swapaxes(label_data, 0, 2).copy()
    label_data = np.rollaxis(label_data, 1, 0).copy()

    # save label with same properties as the mri image
    label = nibabel.Nifti1Image(label_data, image.affine, header=image.header)
    nibabel.nifti1.save(label, lbl_path)

#    The following functions were copied from the nnU-Net repository:
#       "get_identifiers_from_splitted_files()", and
#       "generate_dataset_json()"
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
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
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
    save_json(json_dict, os.path.join(output_file))


if __name__ == '__main__':
    main()
