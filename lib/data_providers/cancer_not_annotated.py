from data_provider import DataProvider
import os
from paths import Paths
import pandas as pd
import numpy as np
import random
import cv2

class CancerNotAnnotated(DataProvider):
    def __init__(self, cfg):
        DataProvider.__init__(self, cfg)
        # Load training images (path) and labels
        train_path = os.path.join(Paths.data_path, 'cell/labels/train.csv')
        test_path = os.path.join(Paths.data_path, 'cell/labels/test.csv')
        data_type = {
            'image_name': np.str,
            'label': np.int
        }
        self._train_df = pd.read_csv(train_path, dtype=data_type)
        self._test_df = pd.read_csv(test_path, dtype=data_type)
        self._train_list = list(self._train_df.index)
        # random.shuffle(self._train_list)
        self._test_list = list(self._test_df.index)
        self._test_size = len(self._test_list)
        self._train_index = 0
        self._test_index = 0

    @staticmethod
    def _get_batch_ids(id_list, id_index, batch_size):
        if id_index + batch_size <= len(id_list):
            batch_ids = id_list[id_index:(id_index + batch_size)]
        else:
            batch_ids = id_list[id_index:] + id_list[:(batch_size + id_index - len(id_list))]
        next_id_index = (id_index + batch_size) % len(id_list)
        return batch_ids, next_id_index

    def _trans_image(self, label, image_name, phase):
        if label == 1:
            image_dir = 'benign'
        else:
            image_dir = 'malignant'
        image_path = os.path.join(Paths.data_path, 'cell/images', image_dir, image_name)
        assert os.path.exists(image_path), \
            'image {} is not existed'.format(image_path)
        img = cv2.imread(image_path)
        img = img.astype(np.float32)
        if phase == 'train':
            #randomly rotate 0, 90, 180, 270
            rotate = random.randint(0,3) * 90
            height, width, _ = img.shape
            center = (width/2, height/2)
            M = cv2.getRotationMatrix2D(center, rotate, 1)
            img = cv2.warpAffine(img, M, (width, height))

            #randomly reverse
            reverse = random.randint(0,1)
            if reverse == 1:
                img = cv2.flip(img, 1)
        img = cv2.resize(img, (self._cfg.resize_length, self._cfg.resize_length))
        img_mean = img.mean(axis=0).mean(axis=0)
        img -= img_mean
        return img

    def _get_batch_data(self, df, batch_ids, phase):
        batch_data = np.array([
            self._trans_image(df['label'][index_id], df['image_name'][index_id], phase)
            for index_id in batch_ids
        ])
        batch_label = np.array([
            df['label'][index_id] for index_id in batch_ids
        ])
        return batch_data, batch_label

    def next_batch(self, batch_size, phase):
        assert phase in ('train', 'test')
        batch_data = None
        batch_label = None
        if phase == 'train':
            batch_ids, self._train_index = self._get_batch_ids(self._train_list, self._train_index, batch_size)
            batch_data, batch_label = self._get_batch_data(self._train_df, batch_ids, phase)
        elif phase == 'test':
            batch_ids, self._test_index = self._get_batch_ids(self._test_list, self._test_index, batch_size)
            batch_data, batch_label = self._get_batch_data(self._test_df, batch_ids, phase)
        return batch_data, batch_label