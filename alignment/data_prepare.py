import tensorflow as tf
import argparse
from SpmTextEncoder import SpmTextEncoder

parser = argparse.ArgumentParser(description='Bpe')
parser.add_argument('--lang1', type=str, required=True,
                    help='path to lang1')
parser.add_argument('--lang2', type=str, required=True,
                    help='path to lang2')
parser.add_argument('--model1', type=str, required=True,
                    help='path to model1')
parser.add_argument('--model2', type=str, required=True,
                    help='path to model2')
args = parser.parse_args()


def _make_example(ids,
                  feature_name="sources"):
    """Make a tf.train.Example for the problem.

    features[input_feature_name] = input_ids

    Also fills in any other required features with dummy values.

    Args:
        ids: list<int>.
        feature_name:

    Returns:
      tf.train.Example
    """
    features = {
        feature_name:
            tf.train.Feature(int64_list=tf.train.Int64List(value=ids))
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


if __name__ == '__main__':
    lang1_encoder = SpmTextEncoder(args.model1)
    lang2_encoder = SpmTextEncoder(args.model2)

    lang1_text_conn = open(args.lang1, 'r')
    lang2_text_conn = open(args.lang2, 'r')

    with tf.python_io.TFRecordWriter(args.lang1 + ".serialized") as lang1,\
            tf.python_io.TFRecordWriter(args.lang2 + ".serialized") as lang2:
        for src, tgt in zip(lang1_text_conn.readlines(), lang2_text_conn.readlines()):
            lang1_ = _make_example(lang1_encoder.encode(src), "sources").SerializeToString()
            lang2_ = _make_example(lang2_encoder.encode(tgt), "targets").SerializeToString()
            lang1.write(lang1_)
            lang2.write(lang2_)
