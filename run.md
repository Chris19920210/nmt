1. generate text files of token ids
`
python generate_spm_corpus_from_data_dir.py --tgt-model=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.zh.model --src-model=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.en.model --src-file=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/nmt_datasets_final/wmt_enzh_32000k_tok_train.lang1 --tgt-file=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/nmt_datasets_final/wmt_enzh_32000k_tok_train.lang2 --direction=enzh --save-dir=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k --mode=train
`
`
/home/chenrihan/anaconda3/bin/python generate_spm_corpus_from_data_dir.py --tgt-model=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.zh.model --src-model=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/t2t_data_bpe_tok_enzh_same_all_32k/vocab.translatespm_enzh_ai32k.32000.subwords.en.model --src-file=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/nmt_datasets_final/wmt_enzh_32000k_tok_dev.lang1 --tgt-file=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/nmt_datasets_final/wmt_enzh_32000k_tok_dev.lang2 --direction=enzh --save-dir=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k --mode=dev
`


2. run, cd to nmt dir

`
export CUDA_VISIBLE_DEVICES=4
`

to train en->zh
`
nohup /home/chenrihan/anaconda3/bin/python -m nmt.nmt --attention=normed_bahdanau --src=en --tgt=zh --vocab_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/vocab --train_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/train --dev_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/dev --test_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/dev --out_dir=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/enzh_normed_bahdanau_384_4 --num_train_steps=5000000 --steps_per_stats=100 --num_layers=4 --num_units=384 --dropout=0.2 > /home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/gnmt_enzh_384_4.log &
`

to train zh->en
`
nohup /home/chenrihan/anaconda3/bin/python -m nmt.nmt --attention=normed_bahdanau --src=zh --tgt=en --vocab_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/vocab --train_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/train --dev_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/dev --test_prefix=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/dev --out_dir=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/zhen_normed_bahdanau_384_4 --num_train_steps=5000000 --steps_per_stats=100 --num_layers=4 --num_units=384 --dropout=0.2 > /home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/gnmt_zhen_384_4.log &
`


3. generate hparams.pkl for each model

`
/home/chenrihan/anaconda3/bin/python -m alignment.my_nmt --out_dir=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/enzh_normed_bahdanau_384_4/ --inference_input_file=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/dev.en --inference_output_file=infer.zh --hparams_pkl=/home/wudong/s2s/dipml/nmt/enzh_hparams.pkl
`

`
/home/chenrihan/anaconda3/bin/python -m alignment.my_nmt --out_dir=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/zhen_normed_bahdanau_384_4/ --inference_input_file=/home/wudong/s2s/dipml/gnmt_enzh_32k_alignment/spm_32k/dev.zh --inference_output_file=infer.en --hparams_pkl=/home/wudong/s2s/dipml/nmt/zhen_hparams.pkl
`


4. test
modify __main__ part, the path of out_dir and hparams_pkl and run

`
python -m alignment.my_alignment
`
