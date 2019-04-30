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
    print('start end list', start_end_list)
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
    thres = 0.5 / max(enzh_att.shape[-1], enzh_att.shape[-2])
    enzh_att[np.where(enzh_att < thres)] = 0
    zhen_att[np.where(zhen_att < thres)] = 0
    enzh_sum = np.sum(enzh_att, axis=1)
    zhen_sum = np.sum(zhen_att, axis=1)
    enzh_att = np.array([tarr / tavg if tavg > 0 else tarr for tarr, tavg in zip(enzh_att, enzh_sum)])
    zhen_att = np.array([tarr / tavg if tavg > 0 else tarr for tarr, tavg in zip(zhen_att, zhen_sum)])

    # from en to zh
    enzh_dic = {}
    for i in range(le):
        tmp_arr = enzh_att[i]
        if np.sum(tmp_arr) == 0: continue
        cur_sorted = np.sort(tmp_arr)
        # if there is a max value that is much larger than others
        if cur_sorted[-1] - cur_sorted[-2] > 0.2 or cur_sorted[-1] / cur_sorted[-2] > 2:
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
        print(align)
        start, end = min(align), max(align) + 1
        return start, end

    def _substitute(self, src_align_ids, tgt_sub_ids, src_ids, tgt_ids, align_matrix, offset):
        print(offset)
        alignments = get_alignment_from_scores(align_matrix[:, :-1])
        print(alignments)
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
        print(src_align_ids)
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

    src_word = '促性腺激素'
    tgt_sub_word = "demo"
    align_matrices = np.load('/home/wudong/s2s/dipml/nmt/demo_npys/attention_images_1556119889.6543574.npy')[:, :, 1:]
    en_zh = False
    if en_zh:
        src_encoder = SpmTextEncoder("/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/"
                                 "t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.en.model")
        tgt_encoder = SpmTextEncoder("/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/"
                                 "t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.zh.model")
        src_txt = 'Controlled ovulation induction is performed using gonadotrophin injections .'
        tgt_txt = '通过 注射 促性腺激素 进行 控制性 排卵 。'
    else:
        tgt_encoder = SpmTextEncoder("/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/"
                                 "t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.en.model")
        src_encoder = SpmTextEncoder("/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/"
                                 "t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.zh.model")
        tgt_txt = 'Controlled ovulation induction is performed using gonadotrophin injections .'
        src_txt = '通过 注射 促性腺激素 进行 控制性 排卵 。'
    ref_ids = [4, 284, 84, 84, 315, 2, 224, 170, 213, 299, 198, 343, 355, 84, 84, 170, 84, 170]

    '''sids = [1845, 20579, 20883, 20682, 20688, 20631, 20563, 13577, 20557, 21590, 8865, 21473, 20563, 21160, 21118, 21043, 21999, 20677, 20741, 7207, 20563, 20849, 20913, 20849, 20643, 20943, 21998, 20985, 20557, 21315, 21131, 20836, 21233, 21556, 20836, 20997, 20582, 20580, 21136, 12618, 20761, 20886, 14158, 20765, 20735, 20761, 21429, 21177, 20693, 20631, 20765, 20564]
    tids = [277, 20759, 8991, 206, 41, 14, 15653, 384, 20557, 273, 170, 1063, 13, 20570, 20548, 17993, 521, 345, 6107, 4645, 34, 343, 348, 1765, 277, 20689, 20610, 577, 1521, 20557, 20689, 20689, 205, 277, 20689, 2812, 2015, 41, 18560, 20557, 20689, 20689, 3588, 2762, 14, 17453, 6872, 31, 7059, 41, 12345, 6, 20689]
    iids = [1594, 14, 16166, 4, 14, 273, 62, 1063, 571, 8, 37, 20548, 16656, 2115, 4, 14, 125, 405, 133, 20544, 37, 20548, 125, 405, 10200, 5871, 280, 4, 8100, 34, 727, 14, 277, 3472, 1521, 318, 205, 277, 98, 8, 1521, 2015, 6]
    print(src_encoder.decode(sids))
    print(tgt_encoder.decode(tids))
    print(tgt_encoder.decode(iids))'''

    #ezsids = [277, 20576, 9300, 20557, 7, 7770, 20423, 41, 98, 18081, 1779, 85, 13273, 6502, 8, 16028, 94, 512, 7, 3262, 41, 6508, 898, 14, 5278, 1874, 41, 3499, 41, 1794, 551, 18687, 4748, 12989, 260, 6637, 11984, 34, 15316, 5417, 86, 5294, 20548, 41, 13273, 6502, 8, 1287, 5496, 260, 6712, 20667, 20667, 34, 234, 20557, 17145, 20818, 5462, 6, 20689]
    #eztids = [1632, 20557, 20761, 5800, 20968, 9576, 20765, 20563, 20574, 21172, 20581, 20723, 21593, 21083, 20563, 19856, 20606, 20910, 20627, 20557, 20578, 20574, 20585, 20590, 20825, 17520, 20684, 20563, 14849, 20606, 20557, 20586, 20907, 20877, 20694, 20692, 21146, 20625, 6031, 21755, 20781, 21593, 21083, 20678, 20634, 20618, 22525, 23233, 21785, 20563, 20634, 20901, 20586, 20779, 20716, 20609, 10309, 20667, 20586, 20597, 8709, 20818, 12096, 21217, 20586, 20564]
    #eziids = [379, 20541, 23530, 151, 20982, 20789, 6007, 20541, 23504, 379, 20541, 23456, 21572, 4, 947, 2175, 21149, 2175, 21149, 10, 181, 947, 2175, 21149, 2578, 4336, 4, 3270, 10, 146, 20238, 21785, 10, 7736, 11455, 10, 7736, 4797, 336, 3758, 607, 3345, 20781, 6712, 20667, 20667, 3345, 20781, 4, 341, 6712, 20667, 20667, 297, 234, 4, 4123, 20818, 4, 6243, 4]
    #ezsids = [277, 20893, 26, 11, 318, 3698, 372, 3510, 1960, 12044, 20556, 20893, 26, 11, 318, 3908, 372, 12373, 43, 7869, 270, 20885, 20885, 21049, 20686, 2591, 20557, 6768, 3790, 43, 2237, 270, 20818, 20772, 21049, 20686, 53, 1327, 5009, 4973, 1885, 603, 101, 43, 1444, 270, 6031, 21049, 20686, 6, 20587, 5868, 20565, 686, 132, 1089, 43, 630, 234, 270, 20772, 21049, 20686, 3912, 6, 20689]
    #eztids = [6018, 20728, 20590, 20744, 20593, 20634, 21820, 20738, 20798, 20557, 20690, 20606, 5371, 20728, 3825, 20627, 20646, 20652, 20885, 20885, 21049, 17815, 20563, 21497, 21142, 20557, 21804, 20895, 21108, 21367, 23665, 20710, 20818, 20772, 21049, 17815, 20557, 21989, 20728, 20671, 21983, 20764, 20710, 6031, 21049, 17815, 20557, 20678, 21804, 23380, 20710, 21175, 20625, 21142, 20687, 20772, 21049, 17815, 20686, 20564]
    #eziids = [69, 1444, 20885, 1754, 270, 1444, 20885, 21049, 3269, 269, 151, 4, 1444, 21049, 3269, 10, 7368, 5253, 4, 1941, 1444, 1754, 270, 2237, 21049, 269, 1596, 7368, 10, 1615, 20895, 5843, 21381, 270, 626, 1754, 269, 4, 20541, 23240, 21653, 21487, 22371, 270, 234, 1754, 269, 100, 1615, 20895, 20541, 23240, 21653, 270, 234, 1754, 269, 4, 1615, 20895, 20541, 23240, 21653, 9]
    ezsids = [213, 132, 5761, 13, 1862, 94, 417, 132, 43, 9236, 183, 251, 20556]
    eztids = [73, 24328, 21336, 9298, 20557, 20626, 21417, 20704, 20673, 20578, 21014, 22162, 20564]
    eziids = [167, 8955, 245, 69, 7962, 4, 167, 215, 66, 69, 7962, 9]
    print(tgt_encoder.decode(ezsids))
    print(src_encoder.decode(eztids))
    print(src_encoder.decode(eziids))

    teids = [8, 13906, 13, 11287, 416, 48, 721, 48316, 7822, 41813, 13906, 31, 3799, 11, 1657, 38, 6895, 34257, 7]
    tzids = [61, 166, 5, 82, 3283, 7130, 3019, 1081, 4, 1899, 4940, 3]
    print(src_encoder.decode(teids))
    print(tgt_encoder.decode(tzids))

    '''align_matrices = np.array([[
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
            [0.01410285, 0.00107198, 0.10915253, 0.94760001, 0.23157741]]])'''
    #align_matrices = np.load('/home/wudong/s2s/dipml/nmt/demo_npys/attention_images_1556028910.642994.npy') #1.
    print(align_matrices.shape)
    ws = WordSubstitution(src_encoder, tgt_encoder)

    #src_txt = 'Gross tubal disease destroys the cilia , so tubal function will not return even if patency is restored .'  #1.
    #tgt_txt = '输卵管 纤毛 破坏 后 ， 输卵管 功能 很难 恢复 ， 即使 输卵管 被 疏通 后 。' #1.
    print(src_encoder.encode(src_txt))
    print(tgt_encoder.encode(tgt_txt))
    print(tgt_encoder.decode(ref_ids))
    src_ids_list = [src_encoder.encode(src_txt)]
    tgt_ids_list = [tgt_encoder.encode(tgt_txt)]
    offsets = [[0] * len(x) for x in src_ids_list]

    tgt_sentences = ws.substitute([src_word], tgt_sub_word, src_ids_list, tgt_ids_list, align_matrices, offsets)

    for each in tgt_sentences:
        print(tgt_encoder.decode(each))

