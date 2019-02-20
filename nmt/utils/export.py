# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Export pre-trained model."""
import os
import time

import tensorflow as tf

from nmt import attention_model
from nmt import model_helper
from nmt.utils import misc_utils
from nmt import gnmt_model
import json
import argparse


class Exporter(object):
    """Export pre-trained model and serve it by tensorflow/serving.
    """

    def __init__(self, hparams, flags):
        """Construct exporter.

        By default, the hparams can be loaded from the `hparams` file
        which saved in out_dir if you enable save_hparams. So if you want to
        export the model, you just add arguments that needed for exporting.
        Arguments are specified in ``nmt.py`` module.
        Go and check that in ``add_export_arugments()`` function.

        Args:
         hparams: Hyperparameter configurations.
         flags: extra flags used for exporting model.
        """
        self.hparams = hparams
        self._model_dir = self.hparams.out_dir
        self._version_number = int(round(time.time()))

        # Decide a checkpoint path
        ckpt_path = self._get_ckpt_path(flags.ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        self._ckpt_path = ckpt.model_checkpoint_path

        self._export_dir = os.path.join(ckpt_path, 'export', str(self._version_number))

        self._print_params()

    def _print_params(self):
        misc_utils.print_hparams(self.hparams)
        print("Export path      : %s" % self._export_dir)
        print("Model directory  : %s" % self._model_dir)
        print("Checkpoint path  : %s" % self._ckpt_path)
        print("Version number   : %d" % self._version_number)

    def _get_ckpt_path(self, flags_ckpt_path):
        ckpt_path = None
        if flags_ckpt_path:
            ckpt_path = flags_ckpt_path
        else:
            for metric in self.hparams.metrics:
                p = getattr(self.hparams, "best_" + metric + "_dir")
                if os.path.exists(p):
                    if self._has_ckpt_file(p):
                        ckpt_path = p
                    break
        if not ckpt_path:
            ckpt_path = self.hparams.out_dir
        return ckpt_path

    @staticmethod
    def _has_ckpt_file(p):
        for f in os.listdir(p):
            if str(f).endswith(".meta"):
                return True
        return False

    def _create_infer_model(self):
        if (hparams.encoder_type == "gnmt" or
                hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
            model_creator = gnmt_model.GNMTModel
        elif hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        else:
            raise ValueError("Unknown attention architecture %s" %
                             hparams.attention_architecture)

        model = model_helper.create_serving_infer_model(model_creator=model_creator,
                                                        hparams=self.hparams, scope=None)
        return model

    def export(self):
        infer_model = self._create_infer_model()
        with tf.Session(graph=infer_model.graph,
                        config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 1})) as sess:
            inference_inputs = infer_model.graph.get_tensor_by_name('src_placeholder:0')

            saver = infer_model.model.saver
            saver.restore(sess, self._ckpt_path)
            sess.run(tf.tables_initializer())
            # note here. Do not use decode func of model.
            sample_id, scores = infer_model.model.serving_infer()
            sample_id = tf.identity(sample_id, "translation")
            scores = tf.identity(sample_id, "beam_score")

            inference_signature = tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={
                    'input': inference_inputs,
                },
                outputs={
                    'outputs': tf.convert_to_tensor(sample_id),
                    'scores': tf.convert_to_tensor(scores)
                }
            )
            legacy_ini_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder = tf.saved_model.builder.SavedModelBuilder(self._export_dir)
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: inference_signature,
                },
                legacy_init_op=legacy_ini_op,
                clear_devices=True,
                assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
            builder.save(as_text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='remover')
    parser.add_argument('--ckpt-path', type=str, default="./",
                        help='model dir (includes checkpoints and hparams file)')
    parser.add_argument('--beam_width', type=int, default=5,
                        help='beam size')
    parser.add_argument('--length_penalty_weight', type=float, default=0.8,
                        help="length_penalty")
    args = parser.parse_args()

    hparams = tf.contrib.training.HParams()
    for k, v in json.load(open(os.path.join(args.ckpt_path, 'hparams'), 'r')).items():

        hparams.add_hparam(k, v)

    hparams.set_hparam('infer_mode', 'beam_search')
    hparams.set_hparam('beam_width', args.beam_width)
    hparams.set_hparam('length_penalty_weight', args.length_penalty_weight)

    exporter = Exporter(hparams=hparams, flags=args)

    exporter.export()
