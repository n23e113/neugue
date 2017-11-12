# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import data_provider
import model

# a placeholder init function, will use this restore model from checkpoint
def init_assign_fn(sess):
      logging.info('Restoring checkpoint(s)')

def train():
    dataset = data_provider.config_to_slim_dataset()
    prefetch_queue = data_provider.slim_dataset_to_prefetch_queue(dataset, 64)

    face_batch, emoji_batch = prefetch_queue.dequeue()

    face_out, emoji_out, face_coding, emoji_coding, _ = model.build_model(face_batch, emoji_batch)
    loss = model.build_loss(face_batch, face_out, face_coding, emoji_batch, emoji_out, emoji_coding)

    global_step = slim.create_global_step()
    train_op = slim.learning.create_train_op(
        loss, tf.train.AdamOptimizer(), summarize_gradients=True)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    slim.learning.train(
      train_op=train_op,
      logdir="./train_dir",
      graph=loss.graph,
      number_of_steps=100000,
      save_summaries_secs=60,
      trace_every_n_steps=1000,
      save_interval_secs=600,
      startup_delay_steps=startup_delay_steps,
      sync_optimizer=sync_optimizer,
      init_fn=init_assign_fn,
      session_config=session_config)
