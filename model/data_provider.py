# -*- coding: utf-8 -*-
"""
decode batch example from tfrecord

"""

import collections
import os
import tensorflow as tf
from tensorflow.contrib import slim

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset')
DEFAULT_CONFIG = {
    'name': 'celeba',
    'size': 90000,
    'pattern': 'celeba*.tfrecord',
    'face_image_shape': (218, 178, 3),
    'emoji_image_shape': (144, 144, 3),
    'items_to_descriptions': {
        ''
    }
}
ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'
])
DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)

def config_to_slim_dataset(config=None, dataset_dir=None):
#
#Args:
#    config: dataset config
#
#Returns:
#    slim.dataset.Dataset
#
    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR

    if not config:
        config = DEFAULT_CONFIG

    zero = tf.zeros([1], dtype=tf.int64)
    keys_to_features = {
        'image_face/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'image_face/format':
        tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image_emoji/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'image_emoji/format':
        tf.FixedLenFeature((), tf.string, default_value='jpg'),
    }

    items_to_handlers = {
      'image_face':
      slim.tfexample_decoder.Image(
          shape=config['face_image_shape'],
          image_key='image_face/encoded',
          format_key='image_face/format'),
      'image_emoji':
      slim.tfexample_decoder.Image(
          shape=config['emoji_image_shape'],
          image_key='image_emoji/encoded',
          format_key='image_emoji/format')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)

    file_pattern = os.path.join(dataset_dir, config['pattern'])

    return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=config['size'],
      items_to_descriptions=config['items_to_descriptions'])

def slim_dataset_to_prefetch_queue(dataset, batch_size):
#Args:
#    dataset: slim.dataset.Dataset
#    batch_size: batch size
#Returns:
#    slim.prefetch_queue.prefetch_queue contain face image batch tensor and emoji image batch tensor

    shuffle_config = DEFAULT_SHUFFLE_CONFIG

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=True,
        common_queue_capacity=16 * batch_size,
        common_queue_min=2 * batch_size)

    face_image, emoji_image = provider.get(['image_face', 'image_emoji'])

    face_image_batch, emoji_image_batch = tf.train.shuffle_batch(
        [face_image, emoji_image],
        batch_size=batch_size,
        num_threads=shuffle_config.num_batching_threads,
        capacity=shuffle_config.queue_capacity,
        min_after_dequeue=shuffle_config.min_after_dequeue)

    # resize to 224 x 192 (h x w)
    face_image_batch = tf.image.resize_bicubic(face_image_batch, [224, 192])
    # resize to 128 x 128 (h x w)
    emoji_image_batch = tf.image.resize_bicubic(emoji_image_batch, [128, 128])

    return slim.prefetch_queue.prefetch_queue([face_image_batch, emoji_image_batch])
