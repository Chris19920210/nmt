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
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

from nmt.utils import vocab_utils


__all__ = ["BatchedInput", "get_infer_iterator", "get_serving_infer_iterator"]
EOS_ID = 2
SOS_ID = 1
UNK = 0


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer",
                            "source",
                            "target_input",
                            "target_output",
                            "source_sequence_length",
                            "target_sequence_length"))):
    pass


def get_infer_iterator(src_dataset,
                       tgt_dataset,
                       src_vocab_table,
                       tgt_vocab_table,
                       batch_size,
                       sos,
                       eos,
                       src_max_len=None,
                       tgt_max_len=None,
                       use_char_encode=False):
    if use_char_encode:
        src_eos_id = vocab_utils.EOS_CHAR_ID
    else:
        src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values))
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt))
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]))


    # Add in the word counts.
    if use_char_encode:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.reshape(vocab_utils.tokens_to_bytes(src), [-1]),
                              tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))
    else:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                              tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0)))

    if use_char_encode:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in: (
                src, tgt_in,
                tf.to_int32(tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN),
                tf.size(tgt_in)))
    else:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in: (
                src, tgt_in, tf.size(src), tf.size(tgt_in)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(
                tf.TensorShape([None]),#src
                tf.TensorShape([None]),#tgt
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_eos_id,  # src
                tgt_eos_id,
                0,
                0))  # src_len -- unused

    batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_ids, src_seq_len, tgt_seq_len) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_ids,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)


def get_serving_infer_iterator(src_dataset,
                               tgt_dataset,
                               batch_size,
                               src_max_len=None,
                               tgt_max_len=None,
                               use_char_encode=False):
    if use_char_encode:
        src_eos_id = vocab_utils.EOS_CHAR_ID
    else:
        src_eos_id = tf.constant(EOS_ID, tf.int32)

    tgt_sos_id = tf.constant(SOS_ID, tf.int32)
    tgt_eos_id = tf.constant(EOS_ID, tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (decode_example(src, "sources"), decode_example(tgt, "targets"))
    )

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt))
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]))

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0)))

    if use_char_encode:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in: (
                src, tgt_in,
                tf.to_int32(tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN),
                tf.size(tgt_in)))
    else:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in: (
                src, tgt_in, tf.size(src), tf.size(tgt_in)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(
                tf.TensorShape([None]),#src
                tf.TensorShape([None]),#tgt
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_eos_id,  # src
                tgt_eos_id,
                0,
                0))  # src_len -- unused

    batched_dataset = batching_func(src_tgt_dataset)
    src_ids, tgt_ids, src_seq_len, tgt_seq_len = tf.contrib.data.get_single_element(batched_dataset)

    return BatchedInput(
        initializer=None,
        source=src_ids,
        target_input=tgt_ids,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)


def decode_example(serialized_example, field):
    data_field = {
        field: tf.VarLenFeature(tf.int64)
    }

    data_items_to_decoder = {
        field: tf.contrib.slim.tfexample_decoder.Tensor(field)
    }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_field, data_items_to_decoder)

    [decoded] = decoder.decode(serialized_example, items=[field])

    return tf.to_int32(decoded)
