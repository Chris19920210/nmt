from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
import os
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import ops
import argparse
import time


def get_graph_def_from_saved_model(saved_model_dir):
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
            session,
            tags=[tag_constants.SERVING],
            export_dir=saved_model_dir
        )
    return meta_graph_def.graph_def


def get_graph_def_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


def freeze_model(saved_model_dir, output_node_names, output_filename):
    output_graph_filename = output_filename
    initializer_nodes = ''
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags=tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print('graph freezed!')


def optimize_graph(model_dir, graph_filename, transforms, output_names):
    input_names = ['src_placeholder', 'tgt_placeholder']
    if graph_filename is None:
        graph_def = get_graph_def_from_saved_model(model_dir)
    else:
        graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    optimized_graph_def = TransformGraph(
        graph_def,
        input_names,
        output_names,
        transforms)
    tf.train.write_graph(optimized_graph_def,
                         logdir=model_dir,
                         as_text=False,
                         name='optimized_model.pb')
    print('Graph optimized!')


def convert_graph_def_to_saved_model(export_dir, graph_filepath):
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={
                'sources': session.graph.get_tensor_by_name('src_placeholder:0'),
                'targets': session.graph.get_tensor_by_name('tgt_placeholder:0')},
            outputs={'outputs': session.graph.get_tensor_by_name('alignment:0')
                     })
        print('Optimized graph converted to SavedModel!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="opt")
    parser.add_argument('--saved-model', type=str, default=None,
                        help='SavedModel model dir')
    parser.add_argument('--export-dir', type=str, default=None,
                        help="export model")
    parser.add_argument('--freeze-model', type=str, default=None,
                        help='path to freeze-model (one of freeze-model or SavedModel should be none)')
    parser.add_argument('--transforms', type=list, nargs='+', default=[
        'remove_nodes(op=Identity)',
        'merge_duplicate_nodes',
        'strip_unused_nodes',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms',
        'fold_old_batch_norms',
        'remove_attribute(attribute_name=_class)'])
    args = parser.parse_args()

    _version_number = int(round(time.time()))

    _export_dir = os.path.join(args.export_dir, 'export', str(_version_number))

    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)

    if args.saved_model is not None:
        freeze_model(args.saved_model, "alignment", os.path.join(args.export_dir, "frozen_model.pb"))

    optimize_graph(args.export_dir, "frozen_model.pb" if args.freeze_model is None else args.freeze_model,
                   args.transforms, ["alignment"])

    convert_graph_def_to_saved_model(_export_dir, os.path.join(args.export_dir, 'optimized_model.pb'))
