""" from https://github.com/keithito/tacotron """

PAD = 0
START_TOKEN = 2 # '<s>'
END_TOKEN = 3 # '</s>'

phonemes  = ["<pad>", "<unk>", "<s>", "</s>"] + \
            ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', # 74개    14
            'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',  # 22
            'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', # 34
            'EY2', 'F', 'G', 'HH', # 38
            'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', # 47
            'M', 'N', 'NG', 'OW0', 'OW1',# 52
            'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
            'UH0', 'UH1', 'UH2', 'UW',
            'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
_punctuation = ['!', '\'', '(', ')', ',', '.', ':', ';', '?', ' ', '-']   # '!\'(),.:;? -' # 11개
symbols = list(phonemes) + list(_punctuation)
