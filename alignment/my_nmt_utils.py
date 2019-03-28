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

"""Utility functions specifically for NMT."""
from __future__ import print_function

import codecs
import time
import numpy as np
import tensorflow as tf

from nmt.utils import evaluation_utils
from nmt.utils import misc_utils as utils

__all__ = ["decode_and_evaluate", "get_translation"]


def find_max_chain(arr, thres=0.265):
  mlen, midx = 0, -1
  j, k = 0, 0
  while j < len(arr):
    if arr[j] < thres:
      j += 1
      continue
    k = j + 1
    while k < len(arr):
      if arr[k] < thres:
        break
      k += 1
    clen = k - j
    if clen > mlen:
      mlen = clen
      midx = j
    elif clen == mlen and sum(arr[midx:midx+mlen]) < sum(arr[j:k]):
      mlen = clen
      midx = j
    j = k + 1
  return midx, mlen


def get_alignment_from_scores(attention_images):
  le, lz = attention_images.shape
  if le <= 1:
    pass
  if lz <= 1:
    pass
  # from en to zh
  enzh_dic = {}
  for i in range(le):
    if len(np.where(attention_images[i, :]<0.08)[0]) == 0:
      continue
    cur_sorted = np.sort(attention_images[i, :])
    # if there are a max value that is much larger than others
    if cur_sorted[-1] / cur_sorted[-2] > 2.2:
      enzh_dic[i] = [np.argmax(attention_images[i, :])]
      continue
    # one to many case
    midx, mlen = find_max_chain(attention_images[i, :])
    if midx != -1:
      enzh_dic[i] = [k for k in range(midx, mlen + midx)]
  print('from en to zh: ', enzh_dic)
  zhen_dic = {}
  for i in range(lz):
    if len(np.where(attention_images[:, i]<0.08)[0]) == 0:
      continue
    cur_sorted = np.sort(attention_images[:, i])
    # if there are a max value that is much larger than others
    if cur_sorted[-1] / cur_sorted[-2] > 2.2:
      zhen_dic[i] = [np.argmax(attention_images[:, i])]
      continue
    # one to many case
    midx, mlen = find_max_chain(attention_images[:, i])
    if midx != -1:
      zhen_dic[i] = [k for k in range(midx, mlen + midx)]
  print('from zh to en: ', zhen_dic)

  alignments = []  
  # check out the alignment with bidirectional confirmation
  for ken in sorted(enzh_dic.keys()):
    zhs = enzh_dic[ken]
    for kzh in zhs:
      if kzh in zhen_dic and ken in zhen_dic[kzh]:
        print(ken, '-', kzh)
        alignments.append([ken, kzh])
  return alignments


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        feed_dict,
                        ref_file,
                        metrics,
                        subword_option,
                        beam_width,
                        tgt_eos,
                        num_translations_per_input=1,
                        decode=True,
                        infer_mode="greedy"):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  if decode:
    utils.print_out("  decoding to output %s" % trans_file)

    start_time = time.time()
    num_sentences = 0

    alignments = []
    attention_images, src_seqlen, trg_seqlen = model.decode(sess, feed_dict)
    print(attention_images.shape, attention_images, src_seqlen, trg_seqlen)
    for i in range(len(attention_images)):
        attention_image = attention_images[i, :src_seqlen[i], :trg_seqlen[i]]
        #print('-----', attention_image.shape, attention_image)
        alignment = get_alignment_from_scores(attention_image)
        alignments.append(alignment)
    return alignments

    '''while True:
      try:
        attention_images = model.decode(sess, feed_dict)
        print('yyyyyyyyyyyyy', attention_images.shape, attention_images)
        tens = 'Conclusion NAT2 M1 mutation genotype was likely associated with the susceptibility to stomach cancer .'.split(' ')
        tzhs = '结论 NAT2M1 基因型 可能 与 胃癌 易感性 有关 。 <end>'.split(' ')
        le, lz = attention_images[0].shape
        print('from en to zh:')
        for i in range(le):
          #idx = np.argmax(attention_images[0, i])
          for idx in range(lz):
            if attention_images[0, i, idx] > 0.1:
              print('\talign', i, tens[i], '->', idx, tzhs[idx], attention_images[0, i, idx])
        print('from zh to en:')
        for i in range(lz):
          #idx = np.argmax(attention_images[0, :, i])
          for idx in range(le):
            if attention_images[0, idx, i] > 0.1:
              print('\talign', i, tzhs[i], '->', idx, tens[idx], attention_images[0, idx, i])
        get_alignment_from_scores(attention_images[:, :, :-1])
        if infer_mode != "beam_search":
          nmt_outputs = np.expand_dims(nmt_outputs, 0)

        batch_size = nmt_outputs.shape[1]
        num_sentences += batch_size

        for sent_id in range(batch_size):
          for beam_id in range(num_translations_per_input):
            translation = get_translation(
                nmt_outputs[beam_id],
                sent_id,
                tgt_eos=tgt_eos,
                subword_option=subword_option)
            trans_f.write((translation + b"\n").decode("utf-8"))
      except tf.errors.OutOfRangeError:
        utils.print_time(
            "  done, num sentences %d" %
            (num_sentences), start_time)
        break'''

  # Evaluation
  '''evaluation_scores = {}
  if ref_file and tf.gfile.Exists(trans_file):
    for metric in metrics:
      score = evaluation_utils.evaluate(
          ref_file,
          trans_file,
          metric,
          subword_option=subword_option)
      evaluation_scores[metric] = score
      utils.print_out("  %s %s: %.1f" % (metric, name, score))

  return evaluation_scores'''


def get_translation(nmt_outputs, sent_id, tgt_eos, subword_option):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]

  if subword_option == "bpe":  # BPE
    translation = utils.format_bpe_text(output)
  elif subword_option == "spm":  # SPM
    translation = utils.format_spm_text(output)
  else:
    translation = utils.format_text(output)

  return translation

