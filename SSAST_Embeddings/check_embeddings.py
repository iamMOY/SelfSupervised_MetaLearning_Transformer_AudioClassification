import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import ASTModel
import numpy as np
from traintest import train, validate
from traintest_mask import trainmask
from reading_wav_file import AudioDatasetIndividual



embeddings_dir = "./embeddings"

classes = os.listdir(embeddings_dir)

num_embeddings = 0

for c in classes:
    num_embeddings += len(os.listdir(os.path.join(embeddings_dir,c)))
    print(os.path.join(embeddings_dir,c),len(os.listdir(os.path.join(embeddings_dir,c))))

print(num_embeddings)