from text import cmudict

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# _arpabet = ['@' + s for s in cmudict.valid_symbols]
# symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

fr_SAMPA = [
    'b', 'd', 'f', 'g', 'H', 'j', 'k', 'l', 'm', 'n',
    'J', 'N', 'p', 'R', 's', 'S', 't', 'v', 'w', 'z',
    'Z', '2', '9', '9~', '@', 'a', 'a~', 'e', 'E', 'e~',
    'i', 'o', 'O', 'o~', 'u', 'y'
]

symbols_fr = [_pad] + list(_special) + list(_punctuation) + fr_SAMPA

_arpabet = ['@' + s for s in cmudict.valid_symbols]

symbols_en = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

symbols_ch = [_pad] + list(_special) + list(_punctuation)
