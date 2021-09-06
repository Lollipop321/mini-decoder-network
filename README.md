# Mini-Decoder Network (MDN)

## Requirements

pytorch >= 1.0, python >= 3.6.0, cuda >= 9.2

## How to Reproduce

To reproduce the experiments, please run:

    # train the wmt14-en2de baseline models
    sh train_en2de_baseline.sh
    # train the wmt14-en2de MDN models
    sh train_en2de_mdn_bpe.sh
    # translate and score on the test sets
    sh translate_en2de.sh

## Implementations

The code files that implements MDN are located in:

`fairseq/models/transformer_mdn.py`