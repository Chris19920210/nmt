# alignment on nmt
remove iterator / vocab / input files

add placeholder of index matrices

# demo
L56 alignment_idx/my_inference,py
```
def inference(ckpt_path,
              hparams,
              scope=None):
```

# placeholders
```
src_index_placeholder = tf.placeholder(shape=[None, None], dtype=tf.int32)

trg_index_placeholder = tf.placeholder(shape=[None, None], dtype=tf.int32)

src_seqlen_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)

trg_seqlen_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)

batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
```

seqlen placeholders are [batch_size] arrays of sequence length of each sentence

# how to run

```
cd /home/wudong/s2s/dipml/nmt-bk

python -m alignment_idx.my_nmt --out_dir=nmt_attention_model --inference_input_file=.
```
