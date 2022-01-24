# -*- coding: utf-8 -*-
# /usr/bin/python

""" from https://github.com/keithito/tacotron """
import re
from text.text_English import cleaners
from text.text_English.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def phoneme_to_sequence(text, cleaner_names):

    sequence = list()
    for t in text:
        if t in _symbol_to_id.keys():
            sequence += [_symbol_to_id[t]]

    return sequence

def phoneme_sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    # result = ''
    result = list()
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result.append(s)

    return result