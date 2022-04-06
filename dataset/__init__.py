# get list of all submodules
from importlib import import_module
from pathlib import Path
import os

__all__ = [
    import_module(f".{f.stem}", __package__)
    for f in Path(__file__).parent.glob("*.py")
    if "__" not in f.stem
]

# retrieve list of available Task modules
module_name = os.path.basename(Path(__file__).parent)
length = len(module_name + ".TaskXXX")
submodule_names = sorted(module.__name__.split('.')[1]
                         for module in __all__
                         if 'Task' in module.__name__
                         and '_' in module.__name__
                         and len(module.__name__.split('_')[0]) is length
                         )

# extract Task numbers
valid_task_numbers = sorted(int(name.split('Task')[1].split('_')[0]) for name in submodule_names)

# create dictionary of task constructors
# Task objects can be dynamically created using 'dataset.make[XXX]'
make = {}
for i in range(0, len(valid_task_numbers)):
    make[valid_task_numbers[i]] = eval(submodule_names[i] + '.Task')
