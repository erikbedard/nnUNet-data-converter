# external libraries
import nibabel
import numpy as np
from dicom2nifti.convert_dicom import dicom_series_to_nifti

# built-in libraries
import os
import sys
import multiprocessing
from statistics import mode

import dataset
import utils


class Task(dataset.Task.Task):
    pass

    def _parse_args(self, parser, args):
        parser.description = "Extract shoulder CT images and labels."
        parser.add_argument(dest="shoulder_dir", type=str,
                            help="Directory where the shoulder data is stored.")
        parser.add_argument("--use_clean_masks", dest="use_clean_masks", default=False, action='store_true',
                            help="Set flag to True to use \"_clean\" masks. "
                                 "Otherwise, the \"_cfill\" masks will be used.")
        args = parser.parse_args(args)

        self.use_clean_masks = args.use_clean_masks
        self._set_input_dir(args.input_dir)


    def print_startup(self):
        print("\n")
        print("##################################")
        print("##        UVic OT&B Lab         ##")
        print("##  Shoulder CT Data Converter  ##")
        print("##################################")
        print("")

    def _set_dataset_info(self):
        description = "Arthritic shoulder CT scans"
        if self.use_clean_masks:
            description += " (clean)"

        self.dataset_info = {
            "dataset_name": "OT&B Lab Shoulder CT",
            "modalities": ["CT"],
            "license": "",
            "description": description,
            "reference": "",
            "release": "0.0",
            "labels": {
                0: "background",
                1: "scapula",
                2: "humerus"}
        }

    def show_user_prompt(self):
        print("CT scans and segmentation masks will be converted from:"
              "\n  \"" + self._input_dir + "\"\n")

    def _set_input_dir(self, input_dir):
        # validate folder structure
        is_valid = True
        shoulder_list = os.listdir(input_dir)

        shoulder_id_list = []
        ct_scan_list = []
        scapula_mask_list = []
        humerus_mask_list = []

        for shoulder_path in shoulder_list:
            exports = os.path.join(input_dir, shoulder_path, 'mimics_exports')
            ct_scan = os.path.join(exports, 'ct_scan')
            shoulder_id = os.path.basename(shoulder_path)
            if self.use_clean_masks:
                scapula_mask = os.path.join(exports, 'scapula_mask_clean')
                humerus_mask = os.path.join(exports, 'humerus_mask_clean')
            else:
                scapula_mask = os.path.join(exports, 'scapula_mask_cfill')
                humerus_mask = os.path.join(exports, 'humerus_mask_cfill')

            if not (  # paths must be non-empty
                    os.path.exists(ct_scan)
                    and os.path.exists(scapula_mask)
                    and os.path.exists(humerus_mask)
                    and len(os.listdir(ct_scan)) > 0
                    and len(os.listdir(scapula_mask)) > 0
                    and len(os.listdir(humerus_mask)) > 0):
                print("WARNING: The following shoulder does not have the required exports:\n  " + shoulder_path)
                is_valid = False

            else:
                shoulder_id_list.append(shoulder_id)
                ct_scan_list.append(ct_scan)
                scapula_mask_list.append(scapula_mask)
                humerus_mask_list.append(humerus_mask)

        if not is_valid:
            sys.exit("ERROR: Invalid input directory. All subdirectories must have the required scan and mask exports.")

        self._input_dir = input_dir
        self._shoulder_id_list = shoulder_id_list
        self._ct_scan_list = ct_scan_list
        self._scapula_mask_list = scapula_mask_list
        self._humerus_mask_list = humerus_mask_list

    def create_images_labels(self, imagesTr_dir, labelsTr_dir):
        # process images
        ct_save_paths = list(os.path.join(imagesTr_dir, shoulder_id + '_0000.nii.gz')
                             for shoulder_id in self._shoulder_id_list)
        utils.convert_dicom_to_nifti(self._ct_scan_list, ct_save_paths)

        # process labels
        _convert_all_masks(self._shoulder_id_list, self._scapula_mask_list, self._humerus_mask_list, labelsTr_dir)


def _convert_all_masks(shoulder_id_list, scapula_mask_list, humerus_mask_list, labelsTr_dir):
    labelsTr_dir_list = [labelsTr_dir] * len(shoulder_id_list)
    data = list(zip(shoulder_id_list, scapula_mask_list, humerus_mask_list, labelsTr_dir_list))

    # process masks
    p = multiprocessing.Pool()
    p.map(_process_masks, data)
    p.close()


def _process_masks(data):
    shoulder_id = data[0]
    scapula_path = data[1]
    humerus_path = data[2]
    save_dir = data[3]

    print("Converting mask: " + shoulder_id)

    # extract masks
    ni_scapula = dicom_series_to_nifti(scapula_path, reorient_nifti=False)['NII']
    scapula = np.array(ni_scapula.dataobj)
    scapula = (scapula != mode(scapula.flatten())).astype('uint8')

    ni_humerus = dicom_series_to_nifti(humerus_path, reorient_nifti=False)['NII']
    humerus = np.array(ni_humerus.dataobj)
    humerus = (humerus != mode(humerus.flatten())).astype('uint8')

    # ensure masks do not overlap, assume scapula is more correct, so modify humerus if needed
    mod_humerus = humerus
    mod_humerus[scapula == humerus] = 0
    if not np.all(humerus == mod_humerus):
        humerus = mod_humerus
        print("Warning: Shoulder \"" + shoulder_id + "\" has overlapping scapula and humerus masks. The humerus mask has been modified.")

    label = scapula + humerus*2

    label_path = os.path.join(save_dir, shoulder_id + ".nii.gz")
    ni_label = nibabel.Nifti1Image(label, ni_scapula.affine, header=ni_scapula.header)
    nibabel.nifti1.save(ni_label, label_path)