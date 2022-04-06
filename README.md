# nnUNet-data-converter
Data converter and formatter for organizing medical images along with their manual segmentations into a format and 
structure that is compatible with nnU-Net (https://github.com/MIC-DKFZ/nnUNet). This tool can be used for:
1. Standardizing an existing dataset
2. Creating single-class data subsets
3. Creating small "toy" versions of datasets for the purpose of quickly running tests on new experimental workflows with nnUNet

Currently, this tool can be used to generate the following datasets:
1. _Task500_TotalKnee_  
Generated from knee MRI data from the Osteoarthritis Initiative (OAI) and manual segmentations by Zuse Institute Berlin (ZIB)
2. _Task600_TotalShoulder_  
Generated from a private shoulder CT dataset created by the Orthopaedic Technologies & Biomechanics 
Lab at the University of Victoria

## Installation
1. Download the repository: \
`git clone https://github.com/erikbedard/nnUNet-data-converter.git`
2. Install the required packages listed in "requirements.txt", e.g.: \
`pip install -r requirements.txt`

## Usage

### Task500_TotalKnee

#### Pre-requisites
To convert data for _Task500_, you must first:
1. Download the OAI baseline data from https://nda.nih.gov/oai/ and unzip the file (e.g. to a directory such as "~/Downloads/Package_1198790/results/00m")
2. Download the OAI-ZIB data of 507 manual segmentations from https://pubdata.zib.de/ and unzip the file (e.g. to a directory such as "~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations")

#### Data Conversion
Use the following command syntax to convert the data: \
``` python data_converter.py 500 SAVE_DIR OAI_DIR OAI_ZIB_DIR```

Where
* `SAVE_DIR` is the directory you want to save the converted data to
* `OAI_DIR` is where the OAI data is located; this directory must contain the subdirectories "OAI_DIR/0.C.2" and "OAI_DIR/0.E.1")
* `OAI_ZIB_DIR` is where the OAI-ZIB data is stored; this directory must contain the subdirectory "OAI_ZIB_DIR/segmentation"


Example:\
```python convert.py 500 "~/nnUNet_raw_data_base/nnUNet_raw_data/" "~/Downloads/Package_1198790/results/00m" "~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations"```

### Task600_TotalShoulder

#### Pre-requisites
To convert data for `Task600`, you must first ensure your shoulder CT and manual segmentation data is organized into 
the following folder structure:
```
SHOULDER_DIR
├── shoulder_id_000
│   └── mimics_exports
│       ├── ct_scan
│       ├── humerus_mask_cfill  
│       ├── humerus_mask_clean
│       ├── scapula_mask_cfill
│       └── scapula_mask_clean
│
├── shoulder_id_001
│   └── mimics_exports
│       ├── ct_scan
│       ├── humerus_mask_cfill  
│       ├── humerus_mask_clean
│       ├── scapula_mask_cfill
│       └── scapula_mask_clean
│
├── ...
```
Where the folders "ct_scan", "humerus_mask_cfill", "humerus_mask_clean", "scapula_mask_cfill",
and "scapula_mask_clean" each contain their respective DICOM series files (e.g. 000.dcm, 001.dcm, ...) for that shoulder. 

Note: Two masks (i.e. "cfill" and "clean") exist for each bone in this dataset. The "cfill" masks have osteophytes present, whereas the 
"clean" masks have had osteophytes removed as part of the manual segmentation process.
Also note that the naming convention _shoulder_id_XXX_ is arbitrary and the actual directory names/IDs can be anything.

#### Data Conversion
Use the following command syntax to convert the data: \
``` python convert.py 600 SAVE_DIR SHOULDER_DIR --use_clean_masks```

Where
* `SAVE_DIR` is where you want to save the converted data
* `SHOULDER_DIR` is where the shoulder data is located

If the option `--use_clean_masks` is specified, the "clean" masks will be used. If the option is omitted, the "cfill" masks will be used.

Example:\
``` python convert.py 600 "~/nnUNet_raw_data_base/nnUNet_raw_data/" "~/ShoulderData"```


## Output Folder Structure
After conversion, the converted images and their labels will have the following train/test data structure:
```
SAVE_DIR
├── dataset.json
├── imagesTr
│   ├── volume000_0000.nii.gz 
│   ├── volume001_0000.nii.gz   
│   ├── ...
│
├── imagesTs
│   ├── volume042_0000.nii.gz 
│   ├── volume043_0000.nii.gz   
│   ├── ...
│ 
├── labelsTr
│   ├── volume000.nii.gz
│   ├── volume001.nii.gz
│   ├── ...
│
└── labelsTr
    ├── volume042.nii.gz
    ├── volume043.nii.gz
    ├── ...
```
Note that the naming convention _volumeXXX_ is arbitrary and the actual file names will depend on the input data source.

<!--- TODO: make this an optional argument --->
A data set is created with all segmentation classes. If there is more than one foreground class, single-class data 
subsets are also created.

<!--- TODO: add argument to ignore user prompt --->