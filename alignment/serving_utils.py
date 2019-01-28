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


def _make_example(src_ids,
                  tgt_ids,
                  src_feature_name="sources",
                  tgt_feature_name="targets"):
    """Make a tf.train.Example for the problem.

    features[input_feature_name] = input_ids

    Also fills in any other required features with dummy values.

    Args:
      src_ids: list<int>.
      tgt_ids: list<int>.
      src_feature_name: name of feature for src_ids.

    Returns:
      tf.train.Example
    """
    src_features = {
        src_feature_name:
            tf.train.Feature(int64_list=tf.train.Int64List(value=src_ids))
    }

    tgt_features = {
        tgt_feature_name:
            tf.train.Feature(int64_list=tf.train.Int64List(value=tgt_ids))
    }

    return tf.train.Example(features=tf.train.Features(feature=tgt_features)), \
        tf.train.Example(features=tf.train.Features(feature=src_features))


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
