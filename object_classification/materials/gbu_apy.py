import os
import pandas as pd
from shutil import copy, copyfile
import xml.etree.ElementTree as ET
from PIL import Image

from IPython import embed

raw_path = "datasets/apy"
gbu_path = "gbu/apy"

if not os.path.exists(gbu_path):
    os.makedirs(gbu_path)

image_data = pd.read_csv("../datasets/apy/image_data.csv")

for i, row in image_data.iterrows():
    src_path = os.path.join(raw_path, row["image_path"])
    dst_path = os.path.join(gbu_path, f"{i}.jpg")
    im = Image.open(src_path)
    w, h = im.size
    xmin = max(row["xmin"], 0)
    ymin = max(row["ymin"], 0)
    xmax = min(row["xmax"], w)
    ymax = min(row["ymax"], h)
    if (
        row["image_path"] == "yahoo_test_images/bag_227.jpg"
        or row["image_path"] == "yahoo_test_images/mug_308.jpg"
    ):
        pass
    else:
        im = im.crop((xmin, ymin, xmax, ymax))
    im.save(dst_path, "JPEG")

print("done!")
