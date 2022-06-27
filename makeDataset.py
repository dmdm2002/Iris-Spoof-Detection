import os
import numpy as np
import tensorflow as tf


class MkDataset():

    def __init__(self):
        super(MkDataset, self).__init__()

    def make_path_list(self, path, classes):
        listdir = os.listdir(f'{path}/{classes}')

        for i in range(len(listdir)):
            listdir[i] = f'{path}/{classes}/{listdir[i]}'

        listdir = np.array(listdir)

        return listdir

    def decode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, 3)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224]) / 255.

        return img

    def configure_for_performance(self, ds, cnt, shuffle=False):
        if shuffle==True:
            ds = ds.shuffle(buffer_size=cnt)
            ds = ds.batch(2)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        elif shuffle==False:
            ds = ds.batch(2)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def make_ds(self, file_path):
        path_ds = tf.data.Dataset.from_tensor_slices(file_path)

        images_ds = path_ds.map(self.decode_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #     labels_ds = tf.data.Dataset.from_tensor_slices(labels)

        #     ds = tf.data.Dataset.zip((images_ds, labels_ds))

        return images_ds

    def get_label(self, file_path):
        # 경로를 경로 구성요소 목록으로 변환합니다
        #     parts = tf.strings.split(file_path, os.path.sep)
        labels = []
        for path in (file_path):
            parts = path.split("/")
            label = None
            if parts[-2] == 'fake':
                label = 0
            elif parts[-2] == 'live':
                label = 1

            labels.append(label)
        # 끝에서 두 번째 요소는 클래스 디렉터리입니다.
        return labels