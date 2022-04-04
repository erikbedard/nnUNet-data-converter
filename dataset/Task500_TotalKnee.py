# external libraries
import dicom2nifti
import SimpleITK as sitk
import nibabel
import numpy as np

# built-in libraries
import os
import glob
import sys
import multiprocessing

import dataset


class Task(dataset.Task.Task):
    pass

    def _parse_args(self, parser, args):
        parser.description = "Extract OAI-ZIB images and labels."
        parser.add_argument(dest="oai_dir", type=str,
                            help="Directory where the OIA data is stored, "
                                 "e.g. \"~/Downloads/Package_1198790/results/00m\"")
        parser.add_argument(dest="oai_zib_dir", type=str,
                            help="Directory where OAI-ZIB data is stored, "
                                 "e.g. \"~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations\"")
        args = parser.parse_args(args)

        self._set_oai_dir(args.oai_dir)
        self._set_oai_zib_dir(args.oai_zib_dir)

    def print_startup(self):
        print("\n")
        print("##############################")
        print("##  OIA-ZIB Data Converter  ##")
        print("##############################")
        print("")

    def _set_dataset_info(self):
        self.dataset_info = {
            "task_name": "TotalKnee",
            "dataset_name": "OAI-ZIB",
            "modalities": ["MRI"],
            "license": "",
            "description": "MRIs from the OAI database with manual segmentations by ZIB",
            "reference": "https://nda.nih.gov/oai/ and https://pubdata.zib.de/",
            "release": "0.0",
            "labels": {
                0: "background",
                1: "femoral bone",
                2: "femoral cartilage",
                3: "tibial bone",
                4: "tibial cartilage"}
        }

    def show_user_prompt(self):
        print("MRIs will be copied from OAI directory:"
              "\n  \"" + self.oai_dir + "\"\n")
        print("Segmentation labels will be copied from OAI-ZIB directory:"
              "\n  \"" + self.oai_zib_dir + "\"\n")

    def _set_oai_dir(self, oai_dir):
        # validate
        dir0C2 = os.path.join(oai_dir, "0.C.2")
        dir0E1 = os.path.join(oai_dir, "0.E.1")
        if not (os.path.exists(dir0C2) and os.path.exists(dir0E1)):
            print("The following directories must exist:")
            print(dir0C2)
            print(dir0E1)
            sys.exit("ERROR: Invalid OAI data directory was specified.")

        self.oai_dir = oai_dir

    def _set_oai_zib_dir(self, oai_zib_dir):

        self.oai_mri_paths = os.path.join(oai_zib_dir, "segmentation", "doc", "oai_mri_paths.txt")
        self.oai_zib_masks_dir = os.path.join(oai_zib_dir, "segmentation", "segmentation_masks")

        # validate
        if not (os.path.exists(self.oai_mri_paths) and os.path.exists(self.oai_zib_masks_dir)):
            print("The following file and directory must exist:")
            print(self.oai_mri_paths)
            print(self.oai_zib_masks_dir)
            sys.exit("ERROR: Invalid OAI-ZIB data directory was specified.")

        self.oai_zib_dir = oai_zib_dir

    def create_images_labels(self, imagesTr_dir, labelsTr_dir):
        # create images
        convert_mri_subset(self.oai_dir, self.oai_mri_paths, imagesTr_dir)

        # create labels
        convert_all_masks(self.oai_zib_masks_dir, labelsTr_dir)
        align_images_and_labels(imagesTr_dir, labelsTr_dir)


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
