# coding=utf-8
# 从t2t训练产生的.lang1 .lang2文件中读取并生成nmt格式的训练集
import os
#import my_spm_utils
from SpmTextEncoder import SpmTextEncoder
from mosestokenizer import MosesDetokenizer
import jieba
import argparse
from sentencepiece import SentencePieceProcessor as sp

def generate_gnmt_data(args, moses=None, user_dict='./dicts.txt', approx_vocab_size=32000):
    src_encoder = SpmTextEncoder(args.src_model)
    tgt_encoder = SpmTextEncoder(args.tgt_model)

    src_suffix = 'en' if args.direction == 'enzh' else 'zh'
    tgt_suffix = 'zh' if args.direction == 'enzh' else 'en'
    save_dir = args.save_dir if args.save_dir[-1] != '/' else args.save_dir[:-1]
    gsrc = open(save_dir + '/' + args.mode + '.' + src_suffix, 'w')
    gtgt = open(save_dir + '/' + args.mode + '.' + tgt_suffix, 'w')

    if args.mode == 'train':
        svocab = open(save_dir + '/vocab.' + src_suffix, 'w')
        tvocab = open(save_dir + '/vocab.' + tgt_suffix, 'w')
        svocab.write('<unk>\n<s>\n</s>\n')
        tvocab.write('<unk>\n<s>\n</s>\n')
        for i in range(3, approx_vocab_size):
            svocab.write(str(i) + '\n')
            tvocab.write(str(i) + '\n')
        svocab.close()
        tvocab.close()

    fsrc = open(args.src_file, 'r')
    ftgt = open(args.tgt_file, 'r')
    sline = fsrc.readline().replace('\n', '')
    tline = ftgt.readline().replace('\n', '')
    while sline and tline:
        if moses is not None:
            if args.direction == 'enzh':
                sline = ' '.join(moses(sline))
                tline = ' '.join(jieba.lcut(tline))
            else:
                tline = ' '.join(moses(tline))
                sline = ' '.join(jieba.lcut(sline))
        print(src_encoder.encode(sline))
        gsrc.write(' '.join(map(str, src_encoder.encode(sline))) + '\n')
        gtgt.write(' '.join(map(str, tgt_encoder.encode(tline))) + '\n')
        sline = fsrc.readline().replace('\n', '')
        tline = ftgt.readline().replace('\n', '')
    fsrc.close()
    ftgt.close()
    gsrc.close()
    gtgt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--src-file', type=str, default=None,
                        help='src')
    parser.add_argument('--tgt-file', type=str, default=None,
                        help='tgt')
    parser.add_argument('--src-model', type=str, default=None,
                        help='src encoder')
    parser.add_argument('--tgt-model', type=str, default=None,
                        help='tgt encoder')
    parser.add_argument('--mode', type=str, default='train',
                        help='generate train/dev/test set')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='dir to save')
    parser.add_argument('--need-tokenize', type=bool, default=False,
                        help='need to tokenize')
    parser.add_argument('--direction', type=str, default='enzh',
                        help='direction of translation')
    args = parser.parse_args()

    if args.need_tokenize:
        moses = MosesDetokenizer('en')
        jieba.load_userdict(user_dict)
        generate_gnmt_data(args, moses=moses)
    else:
        generate_gnmt_data(args)
    '''parser.add_argument('--mode', type=str, default='train',
                        help='generate train/dev/test set')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='dir to save')
    parser.add_argument('--src-suffix', type=str, default='en',
                        help='translation source suffix')
    parser.add_argument('--tgt-suffix', type=str, default='zh',
                        help='translation target suffix')

    src_decoder = sp()
    src_decoder.Load(filename=args.src_model)

    tgt_decoder = sp()
    tgt_decoder.Load(filename=args.tgt_model)

    save_dir = args.save_dir if args.save_dir[-1] != '/' else args.save_dir[:-1]
    fsrc = open(save_dir + '/' + args.mode + '.' + args.src_suffix, 'w')
    ftrg = open(save_dir + '/' + args.mode + '.' + args.tgt_suffix, 'w')
    if args.mode == 'train':
        fsvocab = open(save_dir + '/vocab.' + args.src_suffix, 'w')
        ftvocab = open(save_dir + '/vocab.' + args.tgt_suffix, 'w')
'''


    '''
                print(src_decoder.DecodeIds(source.tolist()))
                # print(target_in)
                print(tgt_decoder.DecodeIds(target_in.tolist()))
                print(tgt_decoder.DecodeIds(target_out.tolist()))
                slen = len(source)
                tlen = len(target_in)
                if slen == 0 or tlen == 0:
                    continue
                if source[0] == 1:
                    sline = ''
                else:
                    sline = str(source[0])
                for i in range(1, slen):
                    if source[i] > max_src_id:
                        max_src_id = source[i]
                    if source[i] != 1 and source[i] != 2:
                        sline += ' ' + str(source[i])
                sline += '\n'
                fsrc.write(sline)

                if target_in[0] == 1:
                    tline = ''
                else:
                    tline = str(target_in[0])
                for i in range(1, tlen):
                    if target_in[i] > max_trg_id:
                        max_trg_id = target_in[i]
                    if target_in[i] != 1 and target_in[i] != 2:
                        tline += ' ' + str(target_in[i])
                tline += '\n'
                ftrg.write(tline)
                #print(sline, tline)
                #if i == 0:
                #    print(sline, tline)
    print('===>>>max', max_src_id, max_trg_id)
    for i in range(max_src_id):
        fsvocab.write(str(i) + '\n')
    for i in range(max_trg_id):
        ftvocab.write(str(i) + '\n')
    fsrc.close()
    ftrg.close()
    fsvocab.close()
    ftvocab.close()
    '''
