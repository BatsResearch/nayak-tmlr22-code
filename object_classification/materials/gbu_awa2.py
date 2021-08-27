import os
import pandas as pd
from shutil import copy, copyfile
import xml.etree.ElementTree as ET


raw_path = "datasets/awa2/JPEGImages"
gbu_path = "gbu/awa2"

if not os.path.exists(gbu_path):
    os.makedirs(gbu_path)

image_label = pd.read_csv("../datasets/awa2/image_label.csv")

for i, row in image_label.iterrows():
    if (i % 1000) == 0:
        print(i)
    src_path = os.path.join(raw_path, row["image_path"])
    dst_path = os.path.join(gbu_path, f"{i}.jpg")
    os.symlink(src_path, dst_path)

print("done!")
