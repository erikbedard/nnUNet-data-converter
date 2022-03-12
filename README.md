# oai-zib-format
Data formatter for organizing OAI knee MRI data and OAI-ZIB segmentations into 'images' and 'labels' folders for semenatic segmentation.

This program is used to extract/standardize the knee MRI data from the OAI data set and the segmentation labels from the OAI-ZIB dataset into 'images' and 'labels' directories.
It is assumed that:
1. The OAI baseline data for 4796 subjects has been downloaded from https://nda.nih.gov/oai/ and the zip file been extracted (e.g. to a directory such as '~/Downloads/Package_1198790/results/00m')
2. The OAI-ZIB data of 507 manual segmentations has been downloaded from https://pubdata.zib.de/ and the zip file has been extract (e.g. to a directory such as '~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations')

## Usage
```python oai_zib.py oai_dir oai_zib_dir save_dir```

Example:
```
python oai_zib.py
    "~/Downloads/Package_1198790/results/00m"
    "~/Downloads/Dataset_1__OAI-ZIB_manual_segmentations"
    "~/data/oai-zib"
```

## Folder/file structure
The 507 MRI images and their labels will be saved with the following structure:
```
save_dir
├── images
│   ├── 9001104
│   │   ├── 9001104-0000.dcm
│   │   ├── ...
│   │   └── 9001104-0159.dcm
│   │
│   ├── ...
│   │
│   └── 9996098
│       ├── 9996098-0000.dcm
│       ├── ...
│       └── 9996098-0159.dcm
│
└── labels
    ├── 9001104
    │   ├── 9001104-0000.png
    │   ├── ...
    │   └── 9001104-0159.png
    │
    ├── ...
    │
    └── 9996098
        ├── 9996098-0000.png
        ├── ...
        └── 9996098-0159.png
```
		
## Note
Each segmentation mask contains voxels labeled as follows: \
0 - background \
1 - femoral bone \
2 - femoral cartilage \
3 - tibial bone \
4 - tibial cartilage
