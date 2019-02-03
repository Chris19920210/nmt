from SpmTextEncoder import SpmTextEncoder
import sys
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoder')
    parser.add_argument("--model", type=str, help="path to model(indices)")
    args = parser.parse_args()
    spm = SpmTextEncoder(args.model)

    for line in sys.stdin:
        sys.stdout.write(spm.decode(list(map(int, line.strip().split()))))
        sys.stdout.write("\n")

    """
    cat ~/nmt/sample_idx.txt | python decode.py --model \
     ~/nmt/t2t_data_enzh_encoder/vocab.translatespm_enzh_ai50k.50000.subwords.en.model
    """
