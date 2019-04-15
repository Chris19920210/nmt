import tensorflow as tf
import collections
import argparse
from sentencepiece import SentencePieceProcessor as sp

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


def get_serving_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True,
                 use_char_encode=False):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

#   if use_char_encode:
#     src_eos_id = vocab_utils.EOS_CHAR_ID
#   else:
#     src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

#   tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
#   tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

#   src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    if use_char_encode:
        src_eos_id = 257
    else:
        src_eos_id = tf.constant(EOS_ID, tf.int32)

    tgt_sos_id = tf.constant(SOS_ID, tf.int32)
    tgt_eos_id = tf.constant(EOS_ID, tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))


    src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, random_seed, reshuffle_each_iteration)


    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (decode_example(src, "sources"), decode_example(tgt, "targets")),
        num_parallel_calls=num_parallel_calls
    ).prefetch(output_buffer_size)


    # src_tgt_dataset = src_tgt_dataset.map(
    #     lambda src, tgt: (
    #         tf.string_split([src]).values, tf.string_split([tgt]).values),
    #     num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    # if use_char_encode:
    #     src_tgt_dataset = src_tgt_dataset.map(
    #         lambda src, tgt: (tf.reshape(vocab_utils.tokens_to_bytes(src), [-1]),
    #                       tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
    #         num_parallel_calls=num_parallel_calls)
    # else:
    #     src_tgt_dataset = src_tgt_dataset.map(
    #         lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
    #                       tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
    #         num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Add in sequence lengths.
    if use_char_encode:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in, tgt_out: (
                src, tgt_in, tgt_out,
                tf.to_int32(tf.size(src) / 50),
                tf.size(tgt_in)),
            num_parallel_calls=num_parallel_calls)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in, tgt_out: (
                src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
            num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt_input
                tf.TensorShape([None]),  # tgt_output
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_eos_id,  # src
                tgt_eos_id,  # tgt_input
                tgt_eos_id,  # tgt_output
                0,  # src_len -- unused
                0))  # tgt_len -- unused

    if num_buckets > 1:

        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    #batched_iter = batched_dataset.make_one_shot_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
        tgt_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--src-file', type=str, default=None,
                        help='src')

    parser.add_argument('--tgt-file', type=str, default=None,
                        help='tgt')
    parser.add_argument('--src-model', type=str, default=None,
                        help='src decoder')
    parser.add_argument('--tgt-model', type=str, default=None,
                        help='src decoder')

    #src_file_placeholder = tf.placeholder(tf.string, shape=())
    #tgt_file_placeholder = tf.placeholder(tf.string, shape=())

    args = parser.parse_args()

    #src_dataset = tf.data.TFRecordDataset(src_file_placeholder)
    #tgt_dataset = tf.data.TFRecordDataset(tgt_file_placeholder)

    src_decoder = sp()
    src_decoder.Load(filename=args.src_model)

    tgt_decoder = sp()
    tgt_decoder.Load(filename=args.tgt_model)

    import numpy as np
    print(src_decoder.DecodeIds(np.array([190, 34, 48297, 2125, 3378, 52, 286, 31, 4875, 11, 121, 19, 38, 4306, 7]).tolist()))
    print(src_decoder.DecodeIds(np.array([178, 2090, 120, 338, 69, 6049, 34, 48288, 305, 5, 3331, 83, 5, 1084, 48, 233, 1679, 119, 17238, 7, 84, 6326, 322, 135, 434, 1559, 336, 7808, 120, 5, 17527, 262, 683, 5, 4573, 26482, 533, 31, 3767, 7, 84, 42114, 1024, 38, 11, 23451, 1296, 11, 2010, 759, 1234, 16063, 438, 137, 1677, 82, 266, 119, 14947, 156, 150, 340, 7]).tolist()))
    print(tgt_decoder.DecodeIds(np.array([19, 726, 1170, 14094, 56, 9925, 5357, 4, 6702, 3]).tolist()))
    print(tgt_decoder.DecodeIds(np.array([6, 18, 28, 1212, 14225]).tolist()))

    '''iterator = get_serving_iterator(
        src_dataset,
        tgt_dataset,
        # src_vocab_table,
        # tgt_vocab_table,
        None, None,
        batch_size=10,
        sos=None,
        eos=None,
        random_seed=1,
        num_buckets=4,
        src_max_len=50,
        tgt_max_len=50,
        skip_count=None,
        num_shards=4,
        shard_index=0,
        use_char_encode=False)
    g = tf.get_default_graph()

    with tf.Session(graph=g) as sess:

        sess.run(iterator.initializer, feed_dict={src_file_placeholder: args.src_file,
                                                  tgt_file_placeholder: args.tgt_file})
        for i in range(1):
            print("======================")
            a, b, c = sess.run([iterator.target_input, iterator.target_output, iterator.source])
            print_format = """
            source: {source: s}
            target_in: {target_in:s}
            target_out: {target_out:s}
            """
            for target_in, target_out, source in zip(a, b, c):
                print(source)
                print(src_decoder.DecodeIds(source.tolist()))
                print(target_in)
                print(tgt_decoder.DecodeIds(target_in.tolist()))
                print(tgt_decoder.DecodeIds(target_out.tolist()))'''
