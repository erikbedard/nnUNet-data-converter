# oai-zib-converter
Data converter and formatter for organizing OAI knee MRI data and OAI-ZIB manual segmentations into a format and 
structure that is compatible with nnU-Net (https://github.com/MIC-DKFZ/nnUNet).

To use this tool, you must first:
1. Download the OAI baseline data for 4796 subjects from https://nda.nih.gov/oai/ and unzip the file (e.g. to a directory such as '~/Downloads/Package_1198790/results/00m')
2. Download the OAI-ZIB data of 507 manual segmentations from https://pubdata.zib.de/ and unzip the file (e.g. to a directory such as '~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations')

## Usage
1. Download the repository: \
`git clone https://github.com/erikbedard/oai-zib-converter.git`
2. Install the required packages listed in "requirements.txt", e.g.: \
`pip install -r requirements.txt`
3. From a terminal window, execute the command: \
`python data_converter.py oai_dir oai_zib_dir save_dir`

Example:
```
python data_converter.py 
    "~/Downloads/Package_1198790/results/00m"
    "~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations"
    "~/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Knee
```

## Folder/file structure
After conversion, the images and their labels have the following train/test data structure:
```
save_dir
├── dataset.json
├── imagesTr
│   ├── volume1_0000.nii.gz 
│   ├── volume2_0000.nii.gz   
│   ├── ...
│
├── imagesTs
│   ├── volume42_0000.nii.gz 
│   ├── volume43_0000.nii.gz   
│   ├── ...
│ 
├── labelsTr
│   ├── volume1.nii.gz
│   ├── volume2.nii.gz
│   ├── ...
│
└── labelsTr
    ├── volume42.nii.gz
    ├── volume43.nii.gz
    ├── ...
```
A data set is created with all segmentation classes. If there is more than one foreground class, single-class data 
subsets are also created.
		
## Note
Each segmentation mask contains voxels labeled as follows: \
0 - background \
1 - femoral bone \
2 - femoral cartilage \
3 - tibial bone \
4 - tibial cartilage
