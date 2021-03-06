import abc
from SpmTextEncoder import SpmTextEncoder
import numpy as np


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))
    return results


def get_src_slice(src_align_ids, src_ids, align_matrix):
    start_end_list = find_sub_list(src_align_ids, src_ids)
    #print('start end list', start_end_list)
    if len(start_end_list) != 0:
        ret = list(map(lambda args: align_matrix[args[0]: args[1] + 1, :], start_end_list))
    else:
        ret = list()
    return ret


# --- for alignment by each 2D attention matrix
def find_max_chain(arr, thres=0.25):
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
        elif clen == mlen and sum(arr[midx:midx + mlen]) < sum(arr[j:k]):
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

    # preprocess for attention image
    enzh_att = np.copy(attention_images)
    zhen_att = np.copy(attention_images.T)
    enzh_att[np.where(enzh_att < 0.1)] = 0
    zhen_att[np.where(zhen_att < 0.1)] = 0
    enzh_sum = np.sum(enzh_att, axis=1)
    zhen_sum = np.sum(zhen_att, axis=1)
    enzh_att = np.array([tarr / tavg if np.sum(tavg) > 0 else 0 for tarr, tavg in zip(enzh_att, enzh_sum)])
    zhen_att = np.array([tarr / tavg if np.sum(tavg) > 0 else 0 for tarr, tavg in zip(zhen_att, zhen_sum)])
    #print(enzh_att[:4, :4])
    #print(zhen_att[:4, :4])

    # from en to zh
    enzh_dic = {}
    for i in range(le):
        tmp_arr = enzh_att[i]
        if np.sum(tmp_arr) == 0: continue
        cur_sorted = np.sort(tmp_arr)
        # if there is a max value that is much larger than others
        if len(cur_sorted) == 0:
            continue
        elif len(cur_sorted) == 1:
            enzh_dic[i] = 0
        elif cur_sorted[-1] - cur_sorted[-2] > 0.2 or cur_sorted[-1] / cur_sorted[-2] > 2:
            enzh_dic[i] = [np.argmax(tmp_arr)]
            continue
        # one to many case
        midx, mlen = find_max_chain(tmp_arr)
        if midx != -1:
            enzh_dic[i] = [k for k in range(midx, mlen + midx)]

        '''if len(np.where(attention_images[i, :]<0.1)[0]) == 0:
          continue
        cur_sorted = np.sort(attention_images[:, i])
        # if there is a max value that is much larger than others
        if cur_sorted[-1] / cur_sorted[-2] > 2:
          enzh_dic[i] = [np.argmax(attention_images[i, :])]
          continue
        # one to many case
        midx, mlen = find_max_chain(attention_images[i, :])
        if midx != -1:
          enzh_dic[i] = [k for k in range(midx, mlen + midx)]'''
    # print('from en to zh: ', enzh_dic)
    zhen_dic = {}
    for i in range(lz):
        tmp_arr = zhen_att[i]
        if np.sum(tmp_arr) == 0: continue
        cur_sorted = np.sort(tmp_arr)
        # if there is a max value that is much larger than others
        if cur_sorted[-1] - cur_sorted[-2] > 0.2 or cur_sorted[-1] / cur_sorted[-2] > 2:
            zhen_dic[i] = [np.argmax(tmp_arr)]
            continue
        # one to many case
        midx, mlen = find_max_chain(tmp_arr)
        if midx != -1:
            zhen_dic[i] = [k for k in range(midx, mlen + midx)]

        '''if len(np.where(attention_images[:, i]<0.1)[0]) == 0:
          continue
        cur_sorted = np.sort(attention_images[:, i])
        # if there is a max value that is much larger than others
        if cur_sorted[-1] / cur_sorted[-2] > 2:
          zhen_dic[i] = [np.argmax(attention_images[:, i])]
          continue
        # one to many case
        midx, mlen = find_max_chain(attention_images[:, i])
        if midx != -1:
          zhen_dic[i] = [k for k in range(midx, mlen + midx)]'''
    # print('from zh to en: ', zhen_dic)

    alignments = {}
    # check out the alignment with bidirectional confirmation
    for ken in sorted(enzh_dic.keys()):
        zhs = enzh_dic[ken]
        for kzh in zhs:
            if kzh in zhen_dic and ken in zhen_dic[kzh]:
                # print(ken, '-', kzh)
                # alignments[ken] = kzh
                if ken not in alignments:
                    alignments[ken] = []
                alignments[ken].append(kzh)
    return alignments


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
        return 3, 4

    '''def _substitute_per(self, tgt_ids, word_src_slice):
        start, end = self.word_alignment(word_src_slice)
        tgt_slice_id = tgt_ids[start: end]
        tgt_word = self.tgt_encoder.decode(tgt_slice_id)
        return tgt_word'''

    def _substitute_per(self, src_range, alignments):
        align = []
        for i in range(src_range[0], src_range[1] + 1):
            if i in alignments:
                cur_align = alignments[i]
                for ca in cur_align:
                    if ca not in align:
                        align.append(ca)
        #print(align)
        start, end = min(align), max(align) + 1
        return start, end

    def _substitute(self, src_align_ids, tgt_sub_ids, src_ids, tgt_ids, align_matrix, offset):
        #print(offset)
        alignments = get_alignment_from_scores(align_matrix[:, :-1])
        #print(alignments)
        src_ranges = find_sub_list(src_align_ids, src_ids)
        # word_src_slices = self.get_word_src_slice(src_word, src_ids, align_matrix)
        # print('word_src_slices', word_src_slices, src_ids)
        if len(src_ranges) == 0:
            return tgt_ids
        else:
            for src_range in src_ranges:
                all_miss = True  # check whether this word is aligned with targets
                for i in range(src_range[0], src_range[1] + 1):
                    if i in alignments:
                        all_miss = False
                        break
                if all_miss:
                    continue

                tgt_index = self._substitute_per(src_range, alignments)
                tgt_ids[tgt_index[0] + offset[tgt_index[0]]: tgt_index[1] + offset[tgt_index[0]]] = tgt_sub_ids
                for i in range(len(offset)):
                    if i > tgt_index[0]:
                        offset[i] += len(tgt_sub_ids) - tgt_index[1] + tgt_index[0]
            return tgt_ids

    def substitute(self, src_words, tgt_word, src_ids_list, tgt_ids_list, align_matrices, offsets):
        # 替换词对不一定是单个词，可能被分词拆开，因此直接传分词后的词数组
        src_align_ids = []
        for src_word in src_words:
            src_align_ids += self.src_encoder.encode(src_word)
        tgt_sub_ids = self.tgt_encoder.encode(tgt_word)
        #print(src_align_ids)
        return list(map(lambda args: self._substitute(
            src_align_ids, tgt_sub_ids, args[0], args[1], args[2],
            args[3]), zip(src_ids_list, tgt_ids_list, align_matrices, offsets)))


if __name__ == '__main__':
    """
    For each src ids and tgt ids pair, we have unique corresponding align matrix.
    src_ids is a 1D list (result of src_encoder.encode)
    tgt_ids is a 1D list (result of tgt_encoder.encode)
    align_matrix is a 2D np.array with shape len(src_ids) by len(tgt_ids) + 1(eos token)
    the corresponding batch is (dimension 0 is batch) 
    src_ids_list is a 2D list
    tgt_ids_list is a 2D list
    align_matrices is a 3D np.array
    Example:
    src sentence: 'Multifrequency Multifrequency electromagnetic method'
    src ids: [7045, 111, 12768, 12170, 769]
    src_pieces: ['▁Mult', 'if', 'requency', '▁electromagnetic', '▁method']
    tgt sentence: 多频 电磁 法
    tgt ids: [77, 14668, 5801, 211]
    tgt_pieces: ['▁多', '频', '▁电磁', '▁法']
    substitute two sentences (src word1: method,src word2: method, 
    tgt_word1: 法, tgt_word2: 法 
    sub1: demo1, sub2:demo2)
    """

    src_encoder = SpmTextEncoder("/home/chris/nmt/"
                                 "t2t_data_enzh_encoder/vocab.translatespm_enzh_ai50k.50000.subwords.en.model")
    tgt_encoder = SpmTextEncoder("/home/chris/nmt/"
                                 "t2t_data_enzh_encoder/vocab.translatespm_enzh_ai50k.50000.subwords.zh.model")

    align_matrices = np.array([[
        [0.40774119, 0.00577477, 0.01190836, 0.0128996, 0.45733237],
        [0.16052449, 0.02375918, 0.00348592, 0.00184908, 0.09769257],
        [0.38648733, 0.95352787, 0.02377507, 0.01076588, 0.0883261],
        [0.03114409, 0.01586617, 0.85167813, 0.02688539, 0.1250716],
        [0.01410285, 0.00107198, 0.10915253, 0.94760001, 0.23157741]],
        [
            [0.40774119, 0.00577477, 0.01190836, 0.0128996, 0.45733237],
            [0.16052449, 0.02375918, 0.00348592, 0.00184908, 0.09769257],
            [0.38648733, 0.95352787, 0.02377507, 0.01076588, 0.0883261],
            [0.03114409, 0.01586617, 0.85167813, 0.02688539, 0.1250716],
            [0.01410285, 0.00107198, 0.10915253, 0.94760001, 0.23157741]]])
    ws = WordSubstitution(src_encoder, tgt_encoder)

    src_word = 'Multifrequency'
    tgt_sub_word = "demo"
    src_ids_list = [[7045, 111, 12768, 12170, 769], [7045, 111, 12768, 12768, 12170, 769]]
    tgt_ids_list = [[77, 14668, 5801, 211], [77, 14668, 5801, 211]]
    offsets = [[0] * len(x) for x in src_ids_list]

    tgt_sentences = ws.substitute([src_word], tgt_sub_word, src_ids_list, tgt_ids_list, align_matrices, offsets)

    for each in tgt_sentences:
        print(tgt_encoder.decode(each))

