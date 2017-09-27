# coding=utf-8

import cv2
import os
import pandas as pd
import numpy as np

annotation_path = '/home/dujiajun/thyroid nodule/annotation/malignant tumour.csv'

df = pd.read_csv(annotation_path,
                 dtype={
                     'image_name': np.str,
                     'x1': np.int,
                     'x2': np.int,
                     'y1': np.int,
                     'y2': np.int
                 },
                 index_col='image_name')


image_dir = '/home/dujiajun/thyroid nodule/malignant tumour'
image_list = os.listdir(image_dir)
image_list.sort()

output_dir = '/home/dujiajun/thyroid nodule/draw malignant tumour'

for image_name in image_list:
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    row = df.loc[image_name]
    if row is not None:
        x1 = row['x1']
        x2 = row['x2']
        y1 = row['y1']
        y2 = row['y2']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)



