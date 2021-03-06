# external libraries
import nibabel
import numpy as np
from dicom2nifti.convert_dicom import dicom_series_to_nifti
from scipy import ndimage
import cc3d
import pandas as pd


# built-in libraries
import os
import sys
import multiprocessing
from statistics import mode

import dataset
import utils
from visualize import compare_masks, visualize


class Task(dataset.Task.Task):
    pass

    def _parse_args(self, parser, args):
        parser.description = "Extract shoulder CT images and labels."
        parser.add_argument(dest="shoulder_dir", type=str,
                            help="Directory where the shoulder data is stored.")
        parser.add_argument("--use_clean_masks", dest="use_clean_masks", default=False, action='store_true',
                            help="Set flag to use \"_clean\" masks. "
                                 "Otherwise, the \"_cfill\" masks will be used.")
        parser.add_argument("--verify_integrity", dest="verify_integrity", default=False, action='store_true',
                            help="Set flag to check that masks have a single connected component "
                                 "and are mutually exclusive.")
        args = parser.parse_args(args)

        self.use_clean_masks = args.use_clean_masks
        self.verify_integrity = args.verify_integrity
        self._set_input_dir(args.shoulder_dir)


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
        _convert_all_masks(self._shoulder_id_list, self._scapula_mask_list, self._humerus_mask_list, labelsTr_dir, self.verify_integrity)


def _convert_all_masks(shoulder_id_list, scapula_mask_list, humerus_mask_list, labelsTr_dir, verify_integrity):
    labelsTr_dir_list = [labelsTr_dir] * len(shoulder_id_list)
    verify_integrity_list = [verify_integrity] * len(shoulder_id_list)
    data = list(zip(shoulder_id_list, scapula_mask_list, humerus_mask_list, labelsTr_dir_list, verify_integrity_list))

    # process masks
    p = multiprocessing.Pool(int(os.cpu_count()/4))
    result = p.map(_process_masks, data)
    p.close()

    # TODO: refactor the saving of JSON files as its own method

    def sort_by_key(dictionary):
        sorted_keys = sorted(dictionary.keys())
        sorted_dict = {}
        for key in sorted_keys:
            sorted_dict[key] = dictionary[key]
        return sorted_dict

    issues_dict_by_id = sort_by_key(dict(result))

    # separate results into separate dicts
    bad_input_dict = {}
    disconnected_dict = {}
    overlap_dict = {}
    holes_dict = {}
    no_issues_list = []
    for data_id, issues in issues_dict_by_id.items():
        if issues:

            # sort issues into separate dicts
            # TODO: the separate dicsts are current broken: they need to allow for multiple descriptions to be saved to
            #  each id (currently only one issue is saved, and saving subsequent issues overwrites what's already there)
            for issue in issues:
                issue_type = issue[0]
                issue_desc = issue[1]
                if issue_type == "BAD INPUT": bad_input_dict[data_id] = issue_desc
                elif issue_type == "DISCONNECTED": disconnected_dict[data_id] = issue_desc
                elif issue_type == "OVERLAP": overlap_dict[data_id] = issue_desc
                elif issue_type == "HOLES": holes_dict[data_id] = issue_desc
                else:
                    raise RuntimeError

        else:
            no_issues_list.append(data_id)

    issues_dict_by_issue = {
        "NO ISSUES": no_issues_list,
        "BAD INPUT": sort_by_key(bad_input_dict),
        "DISCONNECTED": sort_by_key(disconnected_dict),
        "OVERLAP": sort_by_key(overlap_dict),
        "HOLES": sort_by_key(holes_dict)
    }

    summary_dict = {
        "Number shoulders": len(issues_dict_by_id),
        "Number shoulders with issues": len(issues_dict_by_id) - len(no_issues_list),
        "Number of masks": len(issues_dict_by_id) * 2,
        "Number of masks with bad input": len(bad_input_dict),
        "Number of masks with disconnected regions": len(disconnected_dict),
        "Number of masks with holes": len(holes_dict),
        "Number of shoulders with overlapping masks": len(overlap_dict)
    }

    import json
    json_path_by_issue = os.path.join(labelsTr_dir, "data_quality_by_issue.json")
    with open(json_path_by_issue, "w") as file:
        json.dump(issues_dict_by_issue, file, indent=4)

    json_path_by_id = os.path.join(labelsTr_dir, "data_quality_by_id.json")
    with open(json_path_by_id, "w") as file:
        json.dump(issues_dict_by_id, file, indent=4)

    summary_json_path = os.path.join(labelsTr_dir, "data_quality_summary.json")
    with open(summary_json_path, "w") as file:
        json.dump(summary_dict, file, indent=4)


def _process_masks(data):
    shoulder_id = data[0]
    scapula_path = data[1]
    humerus_path = data[2]
    save_dir = data[3]
    verify_integrity = data[4]

    print("Converting mask: " + shoulder_id)

    # extract masks
    ni_scapula = dicom_series_to_nifti(scapula_path, reorient_nifti=False)['NII']
    scapula_input = np.array(ni_scapula.dataobj)
    background_value = mode(scapula_input.flatten())
    scapula = (scapula_input != background_value).astype('uint8')
    scapula_copy = scapula.copy()

    ni_humerus = dicom_series_to_nifti(humerus_path, reorient_nifti=False)['NII']
    humerus_input = np.array(ni_humerus.dataobj)
    background_value = mode(humerus_input.flatten())
    humerus = (humerus_input != background_value).astype('uint8')
    humerus_copy = humerus.copy()

    def calc_regions(mask):

        # structure = np.ones((3, 3, 3))  # 26-connectivity
        # regions, num_regions = ndimage.label(mask, structure=structure)
        regions, num_regions = cc3d.connected_components(mask, return_N=True)

        region_sizes = []
        for i in range(1, num_regions+1):
            region_sizes.append(np.count_nonzero(regions == i))
        return regions, num_regions, region_sizes

    def get_largest_region_and_small_region_frequencies(regions, region_sizes):
        region_sizes_copy = region_sizes.copy()
        largest_region_value_index = np.argmax(region_sizes)
        largest_region_value = largest_region_value_index + 1

        largest_region = (regions == largest_region_value).astype('uint8')

        # report summary of small region sizes
        del region_sizes_copy[largest_region_value_index]
        data = pd.Series(region_sizes_copy)
        small_region_frequencies = dict(data.value_counts(sort=False))

        return largest_region, small_region_frequencies

    warning_list = []
    sep = os.linesep
    if verify_integrity:
        scapula_is_modified = False
        humerus_is_modified = False

        # check that inputs have only two unique label values
        num_labels = len(np.unique(scapula_input))
        if num_labels != 2:
            scapula_is_modified = True
            warning_list.append(
                ("BAD INPUT",
                ["Scapula input mask has " + str(num_labels) + " labels but exactly two were expected (foreground and background).",
                "The largest label (by volume) is assumed to be background and all other labels are assumed to be foreground."])
            )

        num_labels = len(np.unique(humerus_input))
        if num_labels != 2:
            humerus_is_modified = True
            warning_list.append(
                ("BAD INPUT",
                ["Humerus input mask has " + str(num_labels) + " labels but exactly two were expected (foreground and background).",
                "The largest label (by volume) is assumed to be background and all other labels are assumed to be foreground."])
            )

        # ensure masks are a singly connected region
        # check scapula
        regions, num_regions, region_sizes = calc_regions(scapula)
        if num_regions > 1:
            largest_region, small_region_frequencies = get_largest_region_and_small_region_frequencies(regions, region_sizes)
            scapula = largest_region
            scapula_is_modified = True

            warning_list.append(
                ("DISCONNECTED",
                ["Scapula has " + str(num_regions) + " disconnected regions. Only the largest region has been kept.",
                "The smaller disconnected region frequencies are {region_size: frequency}:",
                str(small_region_frequencies)])
            )

        # check humerus
        regions, num_regions, region_sizes = calc_regions(humerus)
        if num_regions > 1:
            largest_region, small_region_frequencies = get_largest_region_and_small_region_frequencies(regions, region_sizes)
            humerus = largest_region
            humerus_is_modified = True

            warning_list.append(
                ("DISCONNECTED",
                ["Humerus has " + str(num_regions) + " disconnected regions. Only the largest region has been kept.",
                "The smaller disconnected region frequencies are {region_size: frequency}:",
                str(small_region_frequencies)])
            )

        # ensure masks do not overlap, assume scapula is more correct, so modify humerus if needed
        overlap = humerus * scapula
        if np.any(overlap):
            humerus[overlap == 1] = 0  # modify humerus to remove overlap
            humerus_is_modified = True

            regions, num_regions, region_sizes = calc_regions(overlap)
            data = pd.Series(region_sizes)
            region_frequencies = dict(data.value_counts(sort=False))

            warning_list.append(
                ("OVERLAP",
                ["Scapula and humerus masks have " + str(num_regions) + " overlapping region(s).",
                "The humerus mask has been modified to remove the overlap.",
                "The overlap region frequencies are {region_size: frequency}:",
                str(region_frequencies)])
            )

        # fill holes
        structure = np.ones((3, 3, 3))  # 26-connectivity
        scapula_filled = ndimage.binary_fill_holes(scapula, structure=structure)
        holes = scapula_filled * (scapula == 0).astype("uint8")
        if np.any(holes == 1):
            scapula = scapula_filled
            scapula_is_modified = True

            regions, num_regions, region_sizes = calc_regions(holes)
            data = pd.Series(region_sizes)
            region_frequencies = dict(data.value_counts(sort=False))

            warning_list.append(
                ("HOLES",
                ["Scapula has " + str(num_regions) + " hole(s). The mask has been modified to fill the hole(s).",
                "The hole region frequencies are {region_size: frequency}:",
                str(region_frequencies)])
            )

        humerus_filled = ndimage.binary_fill_holes(humerus, structure=structure)
        holes = humerus_filled * (humerus == 0).astype("uint8")
        if np.any(holes == 1):
            humerus = humerus_filled
            humerus_is_modified = True

            regions, num_regions, region_sizes = calc_regions(holes)
            data = pd.Series(region_sizes)
            region_frequencies = dict(data.value_counts(sort=False))

            warning_list.append(
                ("HOLES",
                ["Humerus has " + str(num_regions) + " hole(s). The mask has been modified to fill the hole(s).",
                "The hole region frequencies are {region_size: frequency}:",
                str(region_frequencies)])
            )

        print_str = ""
        if warning_list:
            print_str += "WARNING: Shoulder \"" + shoulder_id + "\" has the following issue(s):" + sep
            for item in warning_list:
                type = item[0]
                desc = item[1]
                print_str += "  " + type + ': ' + sep
                for line in desc:
                    print_str += "    " + line + sep
            print(print_str)

    label = scapula + humerus*2

    label_path = os.path.join(save_dir, shoulder_id + ".nii.gz")
    ni_label = nibabel.Nifti1Image(label, ni_scapula.affine, header=ni_scapula.header)
    nibabel.nifti1.save(ni_label, label_path)

    del overlap
    del scapula
    del humerus
    del label
    del ni_label

    if verify_integrity:
        # save original scapula/humerus as temporary files for visualization
        import tempfile
        import time

        if scapula_is_modified:
            scapula_fd, temp_scapula_path = tempfile.mkstemp(suffix=".nii.gz")
            ni_scapula = nibabel.Nifti1Image(scapula_copy, ni_scapula.affine, header=ni_scapula.header)
            nibabel.nifti1.save(ni_scapula, temp_scapula_path)

            del scapula_copy
            del ni_scapula

            # save visualization using CLI
            # TODO: modify visualization code to remove tqdm prompt and replace with simple "Saving visuals" msg
            scapula_video_path = os.path.join(save_dir, shoulder_id + "_scapula_quality.mp4")
            args = [temp_scapula_path, label_path, '--ref_label_num', '1', '--comp_label_num', '1',
                    '--opacities', '0.2', '1', '1', '--video_path', scapula_video_path, '--background']
            visualize.exec_cli(compare_masks.main, args)


            time.sleep(1)
            os.close(scapula_fd)
            time.sleep(1)
            os.remove(temp_scapula_path)

        if humerus_is_modified:
            humerus_fd, temp_humerus_path = tempfile.mkstemp(suffix=".nii.gz")
            ni_humerus = nibabel.Nifti1Image(humerus_copy, ni_humerus.affine, header=ni_humerus.header)
            nibabel.nifti1.save(ni_humerus, temp_humerus_path)

            del humerus_copy
            del ni_humerus

            # save visualization using CLI
            humerus_video_path = os.path.join(save_dir, shoulder_id + "_humerus_quality.mp4")
            args = [temp_humerus_path, label_path, '--ref_label_num', '1', '--comp_label_num', '2',
                    '--opacities', '0.2', '1', '1', '--video_path', humerus_video_path, '--background']
            utils.exec_cli(compare_masks.main, args)

            time.sleep(1)
            os.close(humerus_fd)
            time.sleep(1)
            os.remove(temp_humerus_path)

    return shoulder_id, warning_list