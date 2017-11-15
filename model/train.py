# -*- coding: utf-8 -*-
import logging
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import data_provider
import model

tf.logging.set_verbosity(tf.logging.INFO)

# a placeholder init function, will use this restore model from checkpoint
def init_assign_fn(sess):
      logging.info('Restoring checkpoint(s)')

def train():
    dataset = data_provider.config_to_slim_dataset()
    prefetch_queue = data_provider.slim_dataset_to_prefetch_queue(dataset, 128)

    face_batch, emoji_batch = prefetch_queue.dequeue()
    face_batch = tf.image.convert_image_dtype(face_batch, dtype=tf.float32)
    emoji_batch = tf.image.convert_image_dtype(emoji_batch, dtype=tf.float32)

    face_out, emoji_out, face_coding, emoji_coding, _ = model.build_model(face_batch, emoji_batch)

    tf.summary.image('face_in', face_batch, max_outputs=4)
    tf.summary.image('face_out', face_out, max_outputs=4)

    tf.summary.image('emoji_in', emoji_batch, max_outputs=4)
    tf.summary.image('emoji_out', emoji_out, max_outputs=4)

    total_loss, coding_loss, face_recover_loss, emoji_recover_loss = model.build_loss(
        face_batch, face_out, face_coding, emoji_batch, emoji_out, emoji_coding)
    tf.summary.scalar('coding_Loss', coding_loss)
    tf.summary.scalar('face_recover_loss', face_recover_loss)
    tf.summary.scalar('emoji_recover_loss', emoji_recover_loss)
    tf.summary.scalar('total_loss', total_loss)

    global_step = slim.create_global_step()
    train_op = slim.learning.create_train_op(
        total_loss, tf.train.AdamOptimizer(), summarize_gradients=True)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    slim.learning.train(
      train_op=train_op,
      logdir="./train_dir",
      graph=total_loss.graph,
      number_of_steps=100000,
      save_summaries_secs=60,
      #trace_every_n_steps=1000,
      save_interval_secs=600,
      init_fn=init_assign_fn,
      session_config=session_config)

train()
