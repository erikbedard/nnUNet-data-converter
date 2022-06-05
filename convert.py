import os

import dataset
import utils
import sys
import argparse


def main():

    # startup
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="task_num", type=int,
                        help="Task ID of the dataset.", choices=dataset.valid_task_numbers)
    parser.add_argument(dest="save_dir", type=str,
                        help="Directory where the dataset will be saved, e.g. \"~/data/\". "
                        "A new subdirectory will be created.")
    parser.add_argument("--make_subsets", dest="make_subsets", default=False, action='store_true',
                        help="Set flag to automatically create single-class data subsets.")
    parser.add_argument("--ignore_prompt", dest="ignore_prompt", default=False, action='store_true',
                        help="Set flag to automatically convert data without prompting user for input.")

    # create task object
    args, unknown = parser.parse_known_args()
    all_args = sys.argv[1::]
    task = dataset.make[args.task_num](args.save_dir, parser, all_args)

    make_subsets = args.make_subsets
    ignore_prompt = args.ignore_prompt

    # prompt user
    info = task.dataset_info
    task.print_startup()
    task.show_user_prompt()
    dataset_dir = os.path.join(args.save_dir, task.name)
    print("The dataset will be saved to:\n  \"" + dataset_dir + "\"\n")

    if make_subsets:
        dataset_dir_list = [dataset_dir]
        data_subsets_dirs_dict = utils.get_data_subsets_dirs(dataset_dir, info["labels"])
        if len(data_subsets_dirs_dict) > 0:
            print("The following single-class data subsets will also be created:")
            for val in data_subsets_dirs_dict.values():
                dataset_dir_list.append(val)
                print("  \"" + val + "\"")
    if not ignore_prompt:
        input("\nPress Enter to continue...")

    # generate dataset images and labels
    imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
    labelsTr_dir = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)
    task.create_images_labels(imagesTr_dir, labelsTr_dir)

    # create separate test data
    imagesTs_dir = os.path.join(dataset_dir, "imagesTs")
    utils.split_training_data(imagesTr_dir, labelsTr_dir, train_percent=0.7, seed=0)

    # prepare dataset JSON file
    output_file = os.path.join(dataset_dir, "dataset.json")
    print("\nCreating \"" + output_file + "\"")

    utils.generate_dataset_json(
        output_file, imagesTr_dir, imagesTs_dir,
        info["modalities"], info["labels"], info["dataset_name"],
        info["license"], info["description"],
        info["reference"], info["release"])

    # create new single-class data subsets if applicable
    if make_subsets:
        if len(info["labels"]) > 2:
            utils.make_data_subsets(dataset_dir, info["labels"])

    print("\nFinished!")


if __name__ == '__main__':
    main()
