"""
This module defines dataset-specific functions for reading an existing dataset and converting / consolidating / saving
the dataset's 'image' and 'labels' files into NIFTI format and into two respective directories
"""

import sys
from pathlib import Path
import inspect

from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self, save_dir, parser, args):

        self._set_save_dir(save_dir)
        self._parse_args(parser, args)
        self._set_dataset_info()
        self.name = inspect.getmodule(self).__name__.split('.')[1]

    @abstractmethod
    def print_startup(self): pass

    @abstractmethod
    def show_user_prompt(self): pass

    @abstractmethod
    def create_images_labels(self, imagesTr_dir, labelsTr_dir): pass

    # subclasses must define their own specific set of CLI args needed to perform data conversion
    @abstractmethod
    def _parse_args(self, parser): pass

    @abstractmethod
    def _set_dataset_info(self): pass

    def _set_save_dir(self, save_dir):
        try:
            Path(save_dir).resolve()
        except (OSError, RuntimeError):
            sys.exit("ERROR: Invalid save directory \"" + save_dir + "\"\nSave directory must be a valid path.")
        self.save_dir = save_dir