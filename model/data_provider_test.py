import tensorflow as tf
from tensorflow.python.training import queue_runner_impl
from PIL import Image
import numpy as np

import data_provider

tf.logging.set_verbosity(tf.logging.DEBUG)

def test():
    dataset = data_provider.config_to_slim_dataset()
    prefetch_queue = data_provider.slim_dataset_to_prefetch_queue(dataset, 1)
    face_batch, emoji_batch = prefetch_queue.dequeue()

    init = tf.global_variables_initializer()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        session.run(init)
        for i in xrange(10):

            faces, emojis = session.run([face_batch, emoji_batch])
            im = Image.fromarray(faces[0])
            im.save("{0}_face.jpg".format(i))
            im = Image.fromarray(emojis[0])
            im.save("{0}_emoji.jpg".format(i))

        print("thread.join")
        coord.request_stop()
        coord.join(threads)

test()