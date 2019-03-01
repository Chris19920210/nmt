import abc
from SpmTextEncoder import SpmTextEncoder
import numpy as np


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind+sll-1))
    return results


def get_src_slice(src_align_ids, src_ids, align_matrix):
    start_end_list = find_sub_list(src_align_ids, src_ids)
    if len(start_end_list) != 0:
        ret = list(map(lambda args: align_matrix[args[0]: args[1] + 1, :], start_end_list))
    else:
        ret = list()

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
        return 3, 4

    def _substitute_per(self, tgt_ids, word_src_slice):
        start, end = self.word_alignment(word_src_slice)
        tgt_slice_id = tgt_ids[start: end]
        tgt_word = self.tgt_encoder.decode(tgt_slice_id)
        return tgt_word

    def _substitute(self, src_word, tgt_sub_word, src_ids, tgt_ids, align_matrix):
        word_src_slices = self.get_word_src_slice(src_word, src_ids, align_matrix)
        if len(word_src_slices) == 0:
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

    src sentence: 'Multifrequency electromagnetic method'
    src ids: [7045, 111, 12768, 12170, 769]
    tgt sentence: 多频 电磁 法
    tgt ids: [77, 14668, 5801, 211]



    substitute two sentences (src word: method, tgt_word: 法, sub1: demo1, sub2:demo2)
    """

    src_encoder = SpmTextEncoder("/home/chris/nmt/"
                                 "t2t_data_enzh_encoder/vocab.translatespm_enzh_ai50k.50000.subwords.en.model")
    tgt_encoder = SpmTextEncoder("/home/chris/nmt/"
                                 "t2t_data_enzh_encoder/vocab.translatespm_enzh_ai50k.50000.subwords.zh.model")

    align_matrices = np.array([[[0.40774119, 0.00577477, 0.01190836, 0.0128996, 0.45733237],
                                [0.16052449, 0.02375918, 0.00348592, 0.00184908, 0.09769257],
                                [0.38648733, 0.95352787, 0.02377507, 0.01076588, 0.0883261],
                                [0.03114409, 0.01586617, 0.85167813, 0.02688539, 0.1250716],
                                [0.01410285, 0.00107198, 0.10915253, 0.94760001, 0.23157741]],
                               [[0.40774119, 0.00577477, 0.01190836, 0.0128996, 0.45733237],
                                [0.16052449, 0.02375918, 0.00348592, 0.00184908, 0.09769257],
                                [0.38648733, 0.95352787, 0.02377507, 0.01076588, 0.0883261],
                                [0.03114409, 0.01586617, 0.85167813, 0.02688539, 0.1250716],
                                [0.01410285, 0.00107198, 0.10915253, 0.94760001, 0.23157741]]])
    ws = WordSubstitution(src_encoder, tgt_encoder)

    src_words = ['method', 'method']
    tgt_sub_words = ["demo", "demo2"]
    src_ids_list = [[7045, 111, 12768, 12170, 769], [7045, 111, 12768, 12170, 769]]
    tgt_ids_list = [[77, 14668, 5801, 211], [77, 14668, 5801, 211]]

    tgt_sentences = ws.substitute(src_words, tgt_sub_words, src_ids_list, tgt_ids_list, align_matrices)

    for each in tgt_sentences:
        print(each)
