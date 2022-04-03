# external libraries
import dicom2nifti
import SimpleITK as sitk
import nibabel
import numpy as np

# internal libraries
import sys
import os
import argparse
import glob
import multiprocessing
from pathlib import Path

# local modules
from generate_dataset_json import generate_dataset_json
from split_training_data import split_training_data
from make_data_subsets import make_data_subsets


def main():

    parser = argparse.ArgumentParser(description="Extract OAI-ZIB images and labels.")
    parser.add_argument(dest="oai_dir", type=str,
                        help="Directory where the OIA data is stored, e.g. \"~/Downloads/Package_1198790/results/00m\"")
    parser.add_argument(dest="oai_zib_dir", type=str,
                        help="Directory where OAI-ZIB data is stored, e.g. \"~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations\"")
    parser.add_argument(dest="save_dir", type=str,
                        help="Directory where the dataset will be saved, e.g. \"~/data/\". A new subdirectory will be created.")
    args = parser.parse_args()

    print("\n")
    print("##############################")
    print("##  OIA-ZIB Data Converter  ##")
    print("##############################")
    print("")

    # define label classes and the names of all datasets to be created
    # a new data subset is created for each class
    seg_labels = {
        0: "background",
        1: "femoral bone",
        2: "femoral cartilage",
        3: "tibial bone",
        4: "tibial cartilage"}

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

    # validate save_dir
    try:
        Path(args.save_dir).resolve()
    except (OSError, RuntimeError):
        sys.exit("ERROR: Invalid save directory \"" + args.save_dir + "\"")

    # prompt user
    dataset_dir = os.path.join(args.save_dir, 'Task500_TotalKnee')
    print("OAI MRI scans from:\n  \"" + args.oai_dir + "\"\n"
          "and OAI-ZIB labels from:\n  \"" + args.oai_zib_dir + "\"\n" 
          "will be converted and saved to:\n  \"" + dataset_dir + "\"\n")
    input("Press Enter to continue...")

    # convert MRIs
    imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    convert_mri_subset(args.oai_dir, oai_mri_paths, imagesTr_dir)

    # convert masks
    labelsTr_dir = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(labelsTr_dir, exist_ok=True)
    convert_all_masks(oai_zib_masks_dir, labelsTr_dir)
    align_images_and_labels(imagesTr_dir, labelsTr_dir)

    # create separate test data
    imagesTs_dir = os.path.join(dataset_dir, "imagesTs")
    split_training_data(imagesTr_dir, labelsTr_dir, train_percent=0.7, seed=0)

    # prepare dataset JSON file
    output_file = os.path.join(dataset_dir, "dataset.json")
    print("\nCreating \"" + output_file + "\"")
    modalities = ["MRI"]
    dataset_name = "OAI-ZIB Knee"
    license = ""
    dataset_description = "507 DESS MRIs from the OAI database with manual segmentations provided by ZIB"
    dataset_reference = "https://nda.nih.gov/oai/ and https://pubdata.zib.de/"
    dataset_release = "0.0"

    generate_dataset_json(output_file, imagesTr_dir, imagesTs_dir, modalities,
                          seg_labels, dataset_name, license, dataset_description,
                          dataset_reference, dataset_release)

    make_data_subsets(dataset_dir, seg_labels)

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

    # process masks
    p = multiprocessing.Pool()
    p.map(convert_mhd_to_nifti, data)
    p.close()


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

    # process masks
    p = multiprocessing.Pool()
    p.map(convert_dicom_to_nifti, data)
    p.close()


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
    new_file_name = image_id + '_0000.nii.gz'
    new_file_path = os.path.join(out_dir, new_file_name)
    dicom2nifti.convert_dicom.dicom_series_to_nifti(old_dir, new_file_path, reorient_nifti=False)


def convert_mhd_to_nifti(data):
    image_path = data[0]
    itk_image = sitk.ReadImage(image_path)

    file_name = os.path.basename(image_path)
    image_id = file_name.split('.')[0]  # only keep first part of file name as the image ID

    print("Converting \"" + file_name + "\"")
    save_dir = data[1]
    new_file_name = image_id + '.nii.gz'
    new_file_path = os.path.join(save_dir, new_file_name)

    sitk.WriteImage(itk_image, new_file_path)


def align_images_and_labels(images_dir, labels_dir):
    image_paths = glob.glob(os.path.join(images_dir, '*.nii.gz'))
    image_paths.sort()

    label_paths = glob.glob(os.path.join(labels_dir, '*.nii.gz'))
    label_paths.sort()

    # create data list for parallel processing
    data = zip(image_paths, label_paths)

    # process masks
    p = multiprocessing.Pool()
    p.map(realign, data)
    p.close()


def realign(data):
    im_path = data[0]
    lbl_path = data[1]

    print("Re-aligning: " + lbl_path)

    image = nibabel.nifti1.load(im_path)
    label = nibabel.nifti1.load(lbl_path)

    # re-orient and transform label data so that it properly registers to the image
    label_data = np.array(label.dataobj)
    label_data = np.swapaxes(label_data, 0, 2)
    label_data = np.rollaxis(label_data, 1, 0)
    label_data = np.flip(label_data, axis=1).copy()

    # save label with same properties as the mri image
    label = nibabel.Nifti1Image(label_data, image.affine, header=image.header)
    nibabel.nifti1.save(label, lbl_path)


if __name__ == '__main__':
    main()
