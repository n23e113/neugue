# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def reshape(x, h, w, c):
    x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h*scale, w*scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def face_auto_encoder(scope_name, reuse, image, image_channel, coding_len, convolution_repeat_times, filter_count):
    """
    Args:
        face image -> face image auto encoder
        scope_name: variable scope name
        reuse:
        image: batch image
        image_channel: image channel count, default is 3
        coding_len: auto encoder hidden coding length
        convolution_repeat_times: conv(rev conv) layer count
        filter_count: number of conv filter

    Returns:
        out: image
        coding: auto encoder hidden coding
        variables: trainable variable
    """

    assert(image.shape[1] == 224)
    assert(image.shape[2] == 192)
    assert(image.shape[3] == 3)

    # todo ricker, add batch norm
    # todo ricker, add sigmoid after coding
    print("filter_count:{0} coding_len:{1}".format(filter_count, coding_len))
    print("image shape {0}".format(image.shape))

    with tf.variable_scope(scope_name, reuse=reuse) as vs:
        # encoder
        to_next_layer = slim.conv2d(image, filter_count, 3, 1, activation_fn=tf.nn.elu)

        for idx in range(convolution_repeat_times):
            channel_num = filter_count * (idx + 1)
            to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 1, activation_fn=tf.nn.elu)
            to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < convolution_repeat_times - 1:
                to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 2, activation_fn=tf.nn.elu)
                print("to_next_layer shape {0}".format(to_next_layer.shape))
        print("to_next_layer shape {0}".format(to_next_layer.shape))

        # hidden coding
        to_next_layer = tf.reshape(to_next_layer, [-1, 14 * 12 * channel_num])
        print("to_next_layer shape {0}".format(to_next_layer.shape))
        coding = to_next_layer = slim.fully_connected(to_next_layer, coding_len, activation_fn=None)
        print("coding shape {0}".format(coding.shape))

        # decoder
        num_output = int(np.prod([14, 12, filter_count]))
        to_next_layer = slim.fully_connected(to_next_layer, num_output, activation_fn=None)
        to_next_layer = reshape(to_next_layer, 14, 12, filter_count)
        print("to_next_layer shape {0}".format(to_next_layer.shape))

        for idx in range(convolution_repeat_times):
            to_next_layer = slim.conv2d(to_next_layer, filter_count, 3, 1, activation_fn=tf.nn.elu)
            to_next_layer = slim.conv2d(to_next_layer, filter_count, 3, 1, activation_fn=tf.nn.elu)
            if idx < convolution_repeat_times - 1:
                to_next_layer = upscale(to_next_layer, 2)
                print("to_next_layer shape {0}".format(to_next_layer.shape))
        print("to_next_layer shape {0}".format(to_next_layer.shape))


        out = slim.conv2d(to_next_layer, image_channel, 3, 1, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    print("out shape {0}".format(out.shape))
    return out, coding, variables

def emoji_auto_encoder(scope_name, reuse, image, image_channel, coding_len, convolution_repeat_times, filter_count):
    """
    Args:
        face image -> face image auto encoder
        scope_name: variable scope name
        reuse:
        image: batch image
        image_channel: image channel count, default is 3
        coding_len: auto encoder hidden coding length
        convolution_repeat_times: conv(rev conv) layer count
        filter_count: number of conv filter

    Returns:
        out: image
        coding: auto encoder hidden coding
        variables: trainable variable
    """
    assert(image.shape[1] == 128)
    assert(image.shape[2] == 128)
    assert(image.shape[3] == 3)

    # todo ricker, add batch norm
    # todo ricker, add sigmoid after coding
    print("filter_count:{0} coding_len:{1}".format(filter_count, coding_len))
    print("image shape {0}".format(image.shape))

    with tf.variable_scope(scope_name, reuse=reuse) as vs:
        # encoder
        to_next_layer = slim.conv2d(image, filter_count, 3, 1, activation_fn=tf.nn.elu)

        for idx in range(convolution_repeat_times):
            channel_num = filter_count * (idx + 1)
            to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 1, activation_fn=tf.nn.elu)
            to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < convolution_repeat_times - 1:
                to_next_layer = slim.conv2d(to_next_layer, channel_num, 3, 2, activation_fn=tf.nn.elu)
                print("to_next_layer shape {0}".format(to_next_layer.shape))
        print("to_next_layer shape {0}".format(to_next_layer.shape))

        # hidden coding
        print("to_next_layer shape {0}".format(to_next_layer.shape))
        to_next_layer = tf.reshape(to_next_layer, [-1, 8 * 8 * channel_num])
        print("to_next_layer shape {0}".format(to_next_layer.shape))
        coding = to_next_layer = slim.fully_connected(to_next_layer, coding_len, activation_fn=None)
        print("coding shape {0}".format(coding.shape))

        # decoder
        num_output = int(np.prod([8, 8, filter_count]))
        to_next_layer = slim.fully_connected(to_next_layer, num_output, activation_fn=None)
        to_next_layer = reshape(to_next_layer, 8, 8, filter_count)

        for idx in range(convolution_repeat_times):
            to_next_layer = slim.conv2d(to_next_layer, filter_count, 3, 1, activation_fn=tf.nn.elu)
            to_next_layer = slim.conv2d(to_next_layer, filter_count, 3, 1, activation_fn=tf.nn.elu)
            if idx < convolution_repeat_times - 1:
                to_next_layer = upscale(to_next_layer, 2)
                print("to_next_layer shape {0}".format(to_next_layer.shape))
        print("to_next_layer shape {0}".format(to_next_layer.shape))

        out = slim.conv2d(to_next_layer, image_channel, 3, 1, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    print("out shape {0}".format(out.shape))
    return out, coding, variables

def build_model(face_batch, emoji_batch):
    """
        face_batch: face image batch
        emoji_batch: emoji image batch

        return
        face_out
        emoji_out
        trainable variables
    """
    face_out, face_coding, face_variables = face_auto_encoder("face_auto_encoder",
        False, face_batch, 3, 256, 5, 16)

    emoji_out, emoji_coding, emoji_variables = emoji_auto_encoder("emoji_auto_encoder",
        False, emoji_batch, 3, 64, 5, 8)

    return face_out, emoji_out, face_coding, emoji_coding, (face_variables, emoji_variables)

def build_loss(face_batch, face_out, face_coding, emoji_batch, emoji_out, emoji_coding):
    coding_loss = tf.losses.absolute_difference(emoji_coding[:, 0:48], face_coding[:, 0:48])
    face_recover_loss = tf.losses.absolute_difference(face_batch, face_out)
    emoji_recover_loss = tf.losses.absolute_difference(emoji_batch, emoji_out)
    return coding_loss + face_recover_loss + emoji_recover_loss, coding_loss, face_recover_loss, emoji_recover_loss
