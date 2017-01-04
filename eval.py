#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import json

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("datasample_file_1", "./data/datasample1", "Data source for data sample 1.")
tf.flags.DEFINE_string("datasample_file_2", "./data/datasample2", "Data source for data sample 2.")
tf.flags.DEFINE_string("datasample_file_3", "./data/datasample3", "Data source for data sample 3.")
tf.flags.DEFINE_string("datasample_file_4", "./data/datasample4", "Data source for data sample 4.")
tf.flags.DEFINE_string("datasample_file_5", "./data/datasample5", "Data source for data sample 5.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Evaluated Text
tf.flags.DEFINE_string("evaluated_text", "", "Evaluated Text")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# CHANGE THIS: Load data. Load your own data here
# if FLAGS.eval_train:
#    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.datasample_file_1, FLAGS.datasample_file_2, FLAGS.datasample_file_3, FLAGS.datasample_file_4, FLAGS.datasample_file_5)
#    y_test = np.argmax(y_test, axis=1)
# else:
#    x_raw = ["a masterpiece four years in the making", "everything is off."]
#    y_test = [1, 0]

x_raw = [FLAGS.evaluated_text]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# print("\nEvaluating...\n")

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
        probabilities = graph.get_operation_by_name("output/probabilities").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            batch_probabilities = np.amax(np.around(sess.run(probabilities, {input_x: x_test_batch, dropout_keep_prob: 1.0}), decimals=10), axis=1)
            all_probabilities = np.concatenate([all_probabilities, batch_probabilities])

# Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test))
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), [x + 1 for x in all_predictions], all_probabilities))
print(json.dumps({ "text": predictions_human_readable[0][0], "predictedClass": int(float(predictions_human_readable[0][1])), "probability": float(predictions_human_readable[0][2]) }, ensure_ascii=False, sort_keys=True));
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)