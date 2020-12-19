import tensorflow as tf
import numpy as np

# TODO 모델 경로 변경
tf.flags.DEFINE_string("checkpoint_dir", "./model/1608033392/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()


label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_result(image):
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
            input_x = graph.get_operation_by_name("X").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]

            hypothesis = graph.get_operation_by_name("hypothesis").outputs[0]

            img_hypothesis = sess.run(hypothesis, {input_x: [image], dropout_keep_prob: 1.0})
            result = np.argmax(img_hypothesis[0])
            return label[result]
