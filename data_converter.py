import os
import dataset
import utils


def main():
    # start program, validate inputs, prompt user
    info = dataset.dataset_info()
    args = dataset.parse_args()
    dataset.print_startup()
    dataset.validate_args(args)

    dataset_dir = "Task" + str(args.task_num) + "_" + info["task_name"]
    dataset_dir = os.path.join(args.save_dir, dataset_dir)

    print("Dataset will be saved to:\n  \"" + dataset_dir + "\"\n")
    input("Press Enter to continue...")

    # generate dataset images and labels
    imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
    labelsTr_dir = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)
    dataset.create_images_labels(imagesTr_dir, labelsTr_dir, args)

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

    utils.make_data_subsets(dataset_dir, info["labels"])

    print("Finished!")


if __name__ == '__main__':
    main()
