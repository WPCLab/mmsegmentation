import os
import argparse
import multiprocessing
import immutabledict

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from tqdm import tqdm

from waymo_open_dataset.utils import camera_segmentation_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


class WaymoParser(object):
    def __init__(self,
                 tfrecord_list_file,
                 save_dir,
                 num_workers,
                 test_mode=False):
        self.tfrecord_list_file = tfrecord_list_file
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.test_mode = test_mode

        with open(self.tfrecord_list_file, 'r') as fp:
            self.tfrecord_pathnames = fp.read().splitlines()

        self.image_save_dir = f'{self.save_dir}/images'
        self.label_save_dir = f'{self.save_dir}/labels'

        self.create_folder()

    @staticmethod
    def get_file_id(frame):
        context_name = frame.context.name
        timestamp = frame.timestamp_micros
        file_id = context_name + '-' + str(timestamp) + '-'
        return file_id

    def parse(self):
        print('======Parse Started!======')
        pool = multiprocessing.Pool(self.num_workers)
        gen = list(tqdm(pool.imap(self.parse_one, range(len(self))), total=len(self)))
        pool.close()
        pool.join()
        print('======Parse Finished!======')

    def parse_one(self, index):
        """Convert action for single file.
        Args:
            index (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[index]

        try:
            dataset = tf.data.TFRecordDataset(pathname, compression_type='')
            for frame_idx, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                file_id = self.get_file_id(frame)
                self.save_image_and_label(frame, file_id, frame_idx)

        except Exception as e:
            print('Failed to parse: %s, error msg: %s' % (pathname, str(e)))

        return pathname

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image_and_label(self, frame, file_id, frame_idx):
        """Parse and save the images in png format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_id (str): Current file id.
            frame_idx (int): Current frame index.
        """
        for image_info in frame.images:
            if not image_info.camera_segmentation_label.panoptic_label:
                continue

            # image
            img = tf.image.decode_jpeg(image_info.image)
            img = img.numpy()[...,::-1]

            img_path = f'{self.image_save_dir}/{str(image_info.name - 1)}' + f'{file_id}' + \
                       f'{str(frame_idx).zfill(3)}.png'
            cv2.imwrite(img_path, img)

            # label
            label_info = image_info.camera_segmentation_label
            panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(label_info)
            semantic_label, _ = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                panoptic_label,
                label_info.panoptic_label_divisor
            )

            label_path = f'{self.label_save_dir}/{str(image_info.name - 1)}' + f'{file_id}' + \
                         f'{str(frame_idx).zfill(3)}.png'
            semantic_label = semantic_label.astype(np.uint8)[:, :, 0]

            # convert unlabeled to ignored label (0 to 255)
            semantic_label -= 1
            semantic_label[semantic_label == -1] = 255

            semantic_label = Image.fromarray(semantic_label)
            semantic_label.save(label_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list = [
                self.image_save_dir, self.label_save_dir,
            ]
        else:
            dir_list = [
                self.image_save_dir
            ]
        for d in dir_list:
            if not os.path.exists(d):
                os.makedirs(d)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse waymo image segmentation')
    parser.add_argument(
        '--tfrecord_list_file',
        type=str,
        help='the file with tfrecord file list'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='directory for saving output file'
    )

    parser.add_argument(
        '--test_mode',
        action='store_true'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    parser = WaymoParser(args.tfrecord_list_file, args.save_dir, args.num_workers, args.test_mode)
    parser.parse()