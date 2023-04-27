import os
import numpy as np

spec_dir = ".\\ESC_spec\\"

classes = os.listdir(spec_dir)


num_specs = 0
num_classes = 0
for c in classes:
    num_classes += 1

    spec_files = os.listdir(spec_dir + c)
    num_specs += len(spec_files)

print("Number of classes: ", num_classes)
print("Number of specs: ", num_specs)