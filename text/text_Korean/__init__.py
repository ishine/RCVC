# -*- coding: utf-8 -*-
# /usr/bin/python

import numpy as np
from .korea import ALL_SYMBOLS

_symbol_to_id = {s: i for i, s in enumerate(ALL_SYMBOLS)} # phonemes

def phonemes_to_sequence(pho):

    seq = []
    for ph in pho:
        if ph in _symbol_to_id.keys():
            seq.append(_symbol_to_id[ph])
        else:
            print("error phoneme key!!!!! ", ph)
    return np.array(seq)