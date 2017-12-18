#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("emoji1_data_file", "./test.emoji1", "Data source for the emoji1 data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("emoji0_data_file", "./test.emoji0", "Data source for the emoji0 data.")
tf.flags.DEFINE_string("emoji2_data_file", "./test.emoji2", "Data source for the emoji2 data.")
tf.flags.DEFINE_string("emoji3_data_file", "./test.emoji3", "Data source for the emoji3 data.")
tf.flags.DEFINE_string("emoji4_data_file", "./test.emoji4", "Data source for the emoji4 data.")
tf.flags.DEFINE_string("emoji5_data_file", "./test.emoji5", "Data source for the emoji5 data.")
tf.flags.DEFINE_string("emoji6_data_file", "./test.emoji6", "Data source for the emoji6 data.")
tf.flags.DEFINE_string("emoji7_data_file", "./test.emoji7", "Data source for the emoji7 data.")
tf.flags.DEFINE_string("emoji8_data_file", "./test.emoji8", "Data source for the emoji8 data.")
tf.flags.DEFINE_string("emoji9_data_file", "./test.emoji9", "Data source for the emoji9 data.")
# Eval Parameters
#tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
#    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.emoji1_data_file, FLAGS.emoji0_data_file, FLAGS.emoji2_data_file, FLAGS.emoji3_data_file, FLAGS.emoji4_data_file, FLAGS.emoji5_data_file, FLAGS.emoji6_data_file, FLAGS.emoji7_data_file, FLAGS.emoji8_data_file, FLAGS.emoji9_data_file)
#    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.emoji1_data_file, FLAGS.emoji0_data_file, FLAGS.emoji2_data_file, FLAGS.emoji3_data_file, FLAGS.emoji4_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
     x_raw = ["a masterpiece four years in the making", "everything is off."]
     y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
