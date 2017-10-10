from data_provider import DataProvider
import os
from paths import Paths
import pandas as pd
import numpy as np
import random
import cv2


class CancerAnnotated(DataProvider):
    def __init__(self, cfg):
        DataProvider.__init__(self, cfg)
        csv_dir = os.path.join(Paths.data_path, 'thyroid nodule/annotations', self._cfg.dir_name)
        train_path = os.path.join(csv_dir, 'train.csv')
        test_path = os.path.join(csv_dir, 'test.csv')
        data_type = {
            'image_name': np.str,
            'x1': np.int,
            'y1': np.int,
            'x2': np.int,
            'y2': np.int,
            'label': np.int
        }
        self._train_df = pd.read_csv(train_path, dtype=data_type)
        self._test_df = pd.read_csv(test_path, dtype=data_type)
        self._train_list = list(self._train_df.index)
        random.shuffle(self._train_list)
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

    @staticmethod
    def _apply_margin(x1, y1, x2, y2, width, height, margin_size):
        x_mid = (x1 + x2) // 2
        y_mid = (y1 + y2) // 2
        r1 = (x2 - x1) // 2 + margin_size
        x1 = max(x_mid - r1, 0)
        x2 = min(x_mid + r1, width - 1)
        y1 = max(y_mid - r1, 0)
        y2 = min(y_mid + r1, height - 1)
        r2 = min(min(x_mid - x1, x2 - x_mid), min(y_mid - y1, y2 - y_mid))
        x1 = x_mid - r2
        x2 = x_mid + r2
        y1 = y_mid - r2
        y2 = y_mid + r2
        return x1, y1, x2, y2

    def _crop_image(self, label, image_name, x1, y1, x2, y2, phase):
        if phase == 'train':
            margin_size = random.randint(0, self._cfg.margin_size * 2)
        else:
            margin_size = self._cfg.margin_size
        if label == 0:
            image_dir = 'benign tumour'
        else:
            image_dir = 'malignant tumour'
        image_path = os.path.join(Paths.data_path, 'thyroid nodule/images', image_dir, image_name)
        assert os.path.exists(image_path), \
            'image {} is not existed'.format(image_path)
        img = cv2.imread(image_path)
        img = img.astype(np.float32)
        height, width, _ = img.shape
        x1, y1, x2, y2 = self._apply_margin(x1, y1, x2, y2, width, height, margin_size)
        img = img[y1: y2, x1: x2, :]
        img = cv2.resize(img, (self._cfg.resize_length, self._cfg.resize_length))
        img_mean = img.mean(axis=0).mean(axis=0)
        img -= img_mean
        return img

    def _get_batch_data(self, df, batch_ids, phase):
        batch_data = np.array([
            self._crop_image(df['label'][index_id], df['image_name'][index_id], df['x1'][index_id], df['y1'][index_id], \
                             df['x2'][index_id], df['y2'][index_id], phase)
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

