import csv
import json
import os

import datasets
import pandas as pd
import numpy as np

ds = datasets.load_dataset('wit_dataset_script.py', data_dir='data')
test_ds = ds['validation']

print(test_ds['image_file'])
