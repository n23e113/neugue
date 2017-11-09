import random
import tensorflow as tf
import argparse
import os
import io

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--input_config", help="specify input config file", required=True)
"""
    input config file format
    unicode_point image_dir
    U+1F600 ./grinning_face
    U+1F610 ./neutral_face
"""
args = parser.parse_args()

""" emoji dir contain emoji image file named with unicode point """
g_unicode_emoji_image_path = "./emoji"
g_face_image_height = 218
g_face_image_width = 178
g_emoji_image_height = 144
g_emoji_image_width = 144


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def gen_sample(emoji_file_content, image_path):
    face_file_content = open(image_path, 'r').read()
    
    sample = tf.train.Example(features=tf.train.Features(
    feature={
      'image_face/format': bytes_feature("JPG"),
      'image_face/encoded': bytes_feature(face_file_content),
      'image_face/height': int64_feature(g_face_image_height),
      'image_face/width': int64_feature(g_face_image_width),
      'image_emoji/format': bytes_feature("JPG"),
      'image_emoji/encoded': bytes_feature(emoji_file_content),
      'image_emoji/height': int64_feature(g_emoji_image_height),
      'image_emoji/width': int64_feature(g_emoji_image_width),
    }))
    return sample

def gen_dataset(emoji_path, input_config_filename):
    emoji_files = [f for f in os.listdir(emoji_path) if os.path.isfile(os.path.join(emoji_path, f))]
    # emoji_unicode_point - emoji_file_content
    emoji_files_content = {}
    for file_name in emoji_files:
        content = open(os.path.join(emoji_path, file_name), 'r').read()
        emoji_files_content[os.path.splitext(file_name)[0]] = content

    print(emoji_files_content.keys())

    sample_source = []
    for filelineno, line in enumerate(io.open(input_config_filename, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        # emoji_unicode_point - sample_path
        data = line.split(" ")
        assert(data[0] in emoji_files_content)
        for f in os.listdir(data[1]):
            s = os.path.join(data[1], f)
            if os.path.isfile(s):
                sample_source.append((emoji_files_content[data[0]], s))
    random.shuffle(sample_source)
    
    writer = tf.python_io.TFRecordWriter(os.path.basename(input_config_filename) + ".tfrecord")
    for s in sample_source:
        sample = gen_sample(s[0], s[1])
        writer.write(sample.SerializeToString())
    writer.close()

gen_dataset(g_unicode_emoji_image_path, args.input_config)