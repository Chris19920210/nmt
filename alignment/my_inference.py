# Copyright 2017 Google Inc. All Rights Reserved.
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

"""To perform inference on test set given a trained model."""
from __future__ import print_function

import tensorflow as tf

from . import my_attention_model
from nmt import gnmt_model
from nmt import model_helper
from nmt.utils import misc_utils as utils
from . import my_nmt_utils
from . import my_model_helper

__all__ = ["inference",
           "single_worker_inference"]


def get_model_creator(hparams):
    """Get the right model class depending on configuration."""
    if (hparams.encoder_type == "gnmt" or
            hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
        model_creator = gnmt_model.GNMTModel
    elif hparams.attention_architecture == "standard":
        model_creator = my_attention_model.MyAttentionModel
    else:
        raise ValueError("Unknown attention architecture %s" %
                         hparams.attention_architecture)
    return model_creator


def start_sess_and_load_model(infer_model, ckpt_path):
    """Start session and load model."""
    sess = tf.Session(
        graph=infer_model.graph, config=utils.get_config_proto())
    with infer_model.graph.as_default():
        loaded_infer_model = model_helper.load_model(
            infer_model.model, ckpt_path, sess, "infer")
    return sess, loaded_infer_model


def inference(ckpt_path,
              inference_src_file,
              inference_trg_file,
              inference_output_file,
              hparams,
              scope=None):
    model_creator = get_model_creator(hparams)
    infer_model = my_model_helper.create_infer_model(model_creator, hparams, scope)
    print('===\n', infer_model.src_file_placeholder, infer_model.trg_file_placeholder)
    sess, loaded_infer_model = start_sess_and_load_model(infer_model, ckpt_path)

    single_worker_inference(
        sess,
        infer_model,
        loaded_infer_model,
        inference_src_file,
        inference_trg_file,
        inference_output_file,
        hparams)


def single_worker_inference(sess,
                            infer_model,
                            loaded_infer_model,
                            inference_src_file,
                            inference_trg_file,
                            inference_output_file,
                            hparams):
    """Inference with a single worker."""
    output_infer = inference_output_file

    with infer_model.graph.as_default():
        sess.run(
            infer_model.iterator.initializer,
            feed_dict={
                infer_model.src_file_placeholder: inference_src_file,
                infer_model.trg_file_placeholder: inference_trg_file,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            })
        # Decode
        utils.print_out("# Start decoding")

        my_nmt_utils.decode_and_evaluate(
            "infer",
            loaded_infer_model,
            sess,
            output_infer,
            ref_file=None,
            metrics=hparams.metrics,
            subword_option=hparams.subword_option,
            beam_width=hparams.beam_width,
            tgt_eos=hparams.eos,
            num_translations_per_input=hparams.num_translations_per_input,
            infer_mode=hparams.infer_mode)