from __future__ import print_function

import pickle
import tensorflow as tf

from . import my_attention_model
from . import my_gnmt_model
from nmt import model_helper
from nmt.utils import misc_utils as utils
from . import my_nmt_utils
from . import my_model_helper


def get_model_creator(hparams):
    """Get the right model class depending on configuration."""
    if (hparams.encoder_type == "gnmt" or
            hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
        model_creator = my_gnmt_model.MyGNMTModel
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


class MyAlignment:
    def __init__(self, out_dir, hparams_path, scope=None):
        self.ckpt = tf.train.latest_checkpoint(out_dir)
        self.hparams = pickle.load(open(hparams_path, 'rb'))
        model_creator = get_model_creator(self.hparams)
        self.infer_model = my_model_helper.create_serving_infer_model(model_creator, self.hparams, scope)
        self.sess, self.loaded_infer_model = start_sess_and_load_model(self.infer_model, self.ckpt)
    
    def single_worker_inference(self,
                                inference_src_file,
                                inference_trg_file,
                                # inference_output_file
                                ):
        """Inference with a single worker."""
        output_infer = None

        print("=============", inference_src_file, inference_trg_file)
        with self.infer_model.graph.as_default():
            # Decode
            utils.print_out("# Start decoding")
            feed_dict = {
                self.infer_model.src_file_placeholder: inference_src_file,
                self.infer_model.trg_file_placeholder: inference_trg_file
            }

            return my_nmt_utils.decode_and_evaluate(
                "infer",
                self.loaded_infer_model,
                self.sess,
                output_infer,
                feed_dict,
                ref_file=None,
                metrics=self.hparams.metrics,
                subword_option=self.hparams.subword_option,
                beam_width=self.hparams.beam_width,
                tgt_eos=self.hparams.eos,
                num_translations_per_input=self.hparams.num_translations_per_input,
                infer_mode=self.hparams.infer_mode)

if __name__ == "__main__":
    inference_src_file = [b'\n"\n \n\x07sources\x12\x15\x1a\x13\n\x11\xf4\x04\x00\xa7\x1e\x9a3\xd3\xa6\x01 \xa9\x0e\x99\x07\x19\x07', b'\n\x1e\n\x1c\n\x07sources\x12\x11\x1a\x0f\n\r\xf4\x04\x00\xa7\x1e\x9a3\xd3\xa6\x01 \xa9\x0e', b'\n\x1c\n\x1a\n\x07sources\x12\x0f\x1a\r\n\x0b\xf4\x04\x00\xa7\x1e\x9a3\xd3\xa6\x01 ']
    inference_trg_file = [b'\n\x18\n\x16\n\x07targets\x12\x0b\x1a\t\n\x07\x01\xd2\x02\x00\xb1if', b'\n\x17\n\x15\n\x07targets\x12\n\x1a\x08\n\x06\x01\xd2\x02\x00\xb1i', b'\n\x15\n\x13\n\x07targets\x12\x08\x1a\x06\n\x04\x01\xd2\x02\x00', b'\n\x14\n\x12\n\x07targets\x12\x07\x1a\x05\n\x03\x01\xd2\x02', b'\n\x12\n\x10\n\x07targets\x12\x05\x1a\x03\n\x01\x01']
    aligner = MyAlignment('../new_normed_bahdanau_256', './hparams.pkl')
    print(aligner.single_worker_inference(inference_src_file, inference_trg_file))
