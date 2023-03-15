from text import cmudict, pinyin

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_silences = ["@sp", "@spn", "@sil"]

# _arpabet = ['@' + s for s in cmudict.valid_symbols]
# symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

fr_SAMPA = [
    'b', 'd', 'f', 'g', 'H', 'j', 'k', 'l', 'm', 'n',
    'J', 'N', 'p', 'R', 's', 'S', 't', 'v', 'w', 'z',
    'Z', '2', '9', '9~', '@', 'a', 'a~', 'e', 'E', 'e~',
    'i', 'o', 'O', 'o~', 'u', 'y'
]

fr_IPA = [
    'b','d','f','ɡ','ɥ','j','k','l','m','n',
    'ɲ','ŋ','p','ʁ','s','ʃ','t','v','w','z',
    'ʒ','ø','œ','œ̃','ə','a','ɑ̃','e','ɛ','ɛ̃',
    'i','o','ɔ','ɔ̃','u','y'
]

fr_arpabet = ['@' + s for s in fr_IPA]

symbols_fr = [_pad] + list(_special) + list(_punctuation) + fr_arpabet

_arpabet = ['@' + s for s in cmudict.valid_symbols]

symbols_en = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

_pinyin = ["@" + s for s in pinyin.valid_symbols]

symbols_ch = [_pad] + list(_special) + list(_punctuation) + list(_letters)  + _pinyin + _silences
