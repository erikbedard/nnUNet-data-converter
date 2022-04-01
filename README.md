# oai-zib-converter
Data converter and formatter for organizing OAI knee MRI data and OAI-ZIB manual segmentations into a format and 
structure that is compatible with nnU-Net (https://github.com/MIC-DKFZ/nnUNet).

To use this tool, you must first:
1. Download the OAI baseline data for 4796 subjects from https://nda.nih.gov/oai/ and unzip the file (e.g. to a directory such as '~/Downloads/Package_1198790/results/00m')
2. Download the OAI-ZIB data of 507 manual segmentations from https://pubdata.zib.de/ and unzip the file (e.g. to a directory such as '~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations')

## Usage
```python oai_zib.py oai_dir oai_zib_dir save_dir```

Example:
```
python oai_zib.py
    "~/Downloads/Package_1198790/results/00m"
    "~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations"
    "~/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Knee
```

## Folder/file structure
After conversion, the 507 MRI images and their labels have the following structure:
```
Task500_Knee
├── dataset.json
├── imagesTr
│   ├── knee_9001104_0000.nii.gz 
│   ├── knee_9002430_0000.nii.gz   
│   ├── ...
│   └── knee_9996098_0000.nii.gz
│
└── labelsTr
    ├── knee_9001104.nii.gz
    ├── knee_9002430.nii.gz
    ├── ...
    └── knee_9996098.nii.gz
```
		
## Note
Each segmentation mask contains voxels labeled as follows: \
0 - background \
1 - femoral bone \
2 - femoral cartilage \
3 - tibial bone \
4 - tibial cartilage
