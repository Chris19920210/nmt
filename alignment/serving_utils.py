# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Utilities for serving tensor2tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc

from tensor2tensor.data_generators import text_encoder
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import abc
import numpy as np


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


def _create_stub(server):
    channel = grpc.insecure_channel(server)
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)


def _encode(inputs, encoder, add_eos=True):
    input_ids = encoder.encode(inputs)
    if add_eos:
        input_ids.append(text_encoder.EOS_ID)
    return input_ids


def _decode(output_ids, output_decoder):
    return output_decoder.decode(output_ids, strip_extraneous=True)


def make_grpc_request_fn(servable_name, server, timeout_secs):
    """Wraps function to make grpc requests with runtime args."""
    stub = _create_stub(server)

    def _make_grpc_request(src_examples, tgt_examples):
        """Builds and sends request to TensorFlow model server."""
        request = predict_pb2.PredictRequest()
        request.model_spec.name = servable_name
        assert len(src_examples) == tgt_examples
        request.inputs["sources"].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                [ex.SerializeToString() for ex in src_examples], shape=[len(src_examples)]))

        request.inputs["targets"].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                [ex.SerializeToString() for ex in tgt_examples], shape=[len(tgt_examples)]))

        response = stub.Predict(request, timeout_secs)
        outputs = tf.make_ndarray(response.outputs["outputs"])
        return outputs

    return _make_grpc_request


def predict(src_list, tgt_list, request_fn, src_encoder, tgt_encoder):
    """Encodes inputs, makes request to deployed TF model, and decodes outputs."""
    assert isinstance(src_list, list)
    assert isinstance(tgt_list, list)
    src_ids_list = [
        _encode(src, src_encoder, add_eos=False)
        for src in src_list
    ]
    tgt_ids_list = [
        _encode(tgt, tgt_encoder, add_eos=False)
        for tgt in tgt_list
    ]
    src_examples = [_make_example(src_ids, "sources")
                    for src_ids in src_ids_list]

    tgt_examples = [_make_example(tgt_ids, "targets")
                    for tgt_ids in tgt_ids_list]

    predictions = request_fn(src_examples, tgt_examples)

    return predictions


def indices(mylist, value):
    return [i for i, x in enumerate(mylist) if x == value]


def get_src_slice(src_align_ids, src_ids, align_matrix):
    start_list = indices(src_ids, src_align_ids[0])
    end_list = indices(src_ids, src_align_ids[-1])
    if len(start_list) != 0 and len(end_list) != 0:
        ret = np.array(map(lambda args: align_matrix[args[0]: args[1] + 1, :],
                           zip(start_list, end_list)))
    else:
        ret = np.array([])

    return ret


class WordSubstitution:
    def __init__(self, src_encoder, tgt_encoder):
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder

    def get_word_src_slice(self, src_word, src_ids, align_matrix):
        src_align_ids = self.src_encoder.encode(src_word)
        return get_src_slice(src_align_ids, src_ids, align_matrix)

    @abc.abstractmethod
    def word_alignment(self, word_src_slice):
        """
        :param word_src_slice: the slice of align matrix which src words needs to be substituted
        :return: corresponding start/end indices (a tuple)
        """
        return 0, 1

    def _substitute_per(self, tgt_ids, word_src_slice):
        start, end = self.word_alignment(word_src_slice)
        tgt_slice_id = tgt_ids[start, end]
        tgt_word = self.tgt_encoder.decode(tgt_slice_id)
        return tgt_word

    def _substitute(self, src_word, tgt_sub_word, src_ids, tgt_ids, align_matrix):
        word_src_slices = self.get_word_src_slice(src_word, src_ids, align_matrix)
        if word_src_slices.size == 0:
            return self.tgt_encoder.decode(tgt_ids)
        else:
            tgt_words = map(lambda word_src_slice:
                            self._substitute_per(tgt_ids, word_src_slice), word_src_slices)
            tgt_sentence = self.tgt_encoder.decode(tgt_ids)
            for tgt_word in tgt_words:
                tgt_sentence = tgt_sentence.replace(tgt_word, tgt_sub_word)
            return tgt_sentence

    def substitute(self, src_words, tgt_sub_words, src_ids_list, tgt_ids_list, align_matrices):
        return list(map(lambda args: self._substitute(
            args[0], args[1], args[2], args[3], args[4]
        ), zip(src_words, tgt_sub_words, src_ids_list, tgt_ids_list, align_matrices)))
