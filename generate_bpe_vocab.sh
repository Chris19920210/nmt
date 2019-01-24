SUBWORD_DIR=/home/wudong/s2s/dipml
OUTPUT_DIR=/home/wudong/s2s/dipml/translated_align
# Learn Shared BPE
for merge_ops in 50000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.en" "${OUTPUT_DIR}/train.tok.zh" | \
    ${SUBWORD_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en zh; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${SUBWORD_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  echo -e "<unk>\n<s>\n</s>" > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
  cat "${OUTPUT_DIR}/train.tok.bpe.${merge_ops}.en" "${OUTPUT_DIR}/train.tok.bpe.${merge_ops}.zh" | \
    ${SUBWORD_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

done

# Duplicate vocab file with language suffix
cp "${OUTPUT_DIR}/vocab.bpe.50000" "${OUTPUT_DIR}/vocab.bpe.50000.en"
cp "${OUTPUT_DIR}/vocab.bpe.50000" "${OUTPUT_DIR}/vocab.bpe.50000.zh"

echo "All done."
