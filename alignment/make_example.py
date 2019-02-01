import numpy as np
import tensorflow as tf

source = np.array([[628], [0], [3879], [6554], [21331], [32], [1833], [921], [25], [7], [16241], [11], [5943], [694], [6]])
target_input = np.array([[1], [338], [0], [13489], [102], [57], [11677], [19601], [577], [8]])


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


def make_examples(src, tgt, src_field, tgt_field, num=5):
    src_result = []
    tgt_result = []
    for _ in range(num):
        src_result.append(_make_example(src, src_field).SerializeToString())
        tgt_result.append(_make_example(tgt, tgt_field).SerializeToString())
        src.pop()
        tgt.pop()

    return src_result, tgt_result


source = np.reshape(source, (15)).tolist()
target_input = np.reshape(target_input, (10)).tolist()

src, tgt = make_examples(source, target_input, "sources", "targets")


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "/home/chenrihan/nmt/export_align/1548834206")
    graph = tf.get_default_graph()
    src_placeholder = graph.get_tensor_by_name("src_placeholder:0")
    tgt_placeholder = graph.get_tensor_by_name("tgt_placeholder:0")
    model = graph.get_tensor_by_name("alignment:0")
    a = sess.run(model, {src_placeholder: src, tgt_placeholder: tgt})
    print(a)
    print(type(a))
    print(a.shape)
