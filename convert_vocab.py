#coding=utf-8

#prefix = "/home/wudong/s2s/dipml/t2t_data_enzh_encoder/vocab.translatespm_enzh_ai50k.50000.subwords.vocab"
prefix = "/home/chenrihan/nmt_datasets_spm/t2t_data_enzh_encoder/vocab.translatespm_enzh_ai50k.50000.subwords"

def convert_vocab(prefix, lang):
    dic = {}
    f = open(prefix + '.' + lang + '.vocab', 'rb')
    fres = open('spm_vocab.' + lang, 'wb')
    line = f.readline()
    while line is not None and line != "":
        tok = line.split(b'\t')[0]
        print(line, "|||", tok)
        stok = tok.decode('utf-8')
        if stok in dic:
            print(stok)
        else:
            dic[stok] = 1
        fres.write(tok + b'\n')
        line = f.readline()
    f.close()
    fres.close()

convert_vocab(prefix, 'en')
convert_vocab(prefix, 'zh')
