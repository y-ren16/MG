# import re
# from text import cleaners
import re
from text import cleaners
from text.symbols import symbols_fr, symbols_en, symbols_ch

# _symbol_to_id = {s: i for i, s in enumerate(symbols)}
# _id_to_symbol = {i: s for i, s in enumerate(symbols)}

_symbol_to_id_fr = {s: i for i, s in enumerate(symbols_fr)}
_id_to_symbol_fr = {i: s for i, s in enumerate(symbols_fr)}

_symbol_to_id_en = {s: i for i, s in enumerate(symbols_en)}
_id_to_symbol_en = {i: s for i, s in enumerate(symbols_en)}

_symbol_to_id_ch = {s: i for i, s in enumerate(symbols_ch)}
_id_to_symbol_ch = {i: s for i, s in enumerate(symbols_ch)}

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(language, have_phone, text, cleaner_names, dictionary=None):
    sequence = []

    if language == 'fr':
        # skip = False
        # SAMPA_i = text
        # for j in range(len(SAMPA_i)):
        #     if skip:
        #         skip = False
        #         continue
        #     if j == len(SAMPA_i) - 1:
        #         sequence.append(_symbol_to_id_fr[SAMPA_i[j]])
        #     elif SAMPA_i[j + 1] == '~':
        #         sequence.append(_symbol_to_id_fr[SAMPA_i[j] + '~'])
        #         skip = True
        #     else:
        #         sequence.append(_symbol_to_id_fr[SAMPA_i[j]])
        space = _symbols_to_sequence_fr(' ')
        start = _symbols_to_sequence_fr(['@sp'])
        end = _symbols_to_sequence_fr(['@spn'])
        sequence += start
        for pp in text.split():
            iippaa = []
            for j in pp:
                if j == '̃':
                    iippaa[-1] = iippaa[-1] + j
                elif j == 'ː': 
                    continue   
                else:
                    iippaa.append(j)
            ipa = ' '.join([i for i in iippaa])
            sequence += _arpabet_to_sequence_fr(ipa)
            sequence += space
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
        sequence += end
    if language == 'en':
        space = _symbols_to_sequence_en(' ')
        start = _symbols_to_sequence_en(['@sp'])
        end = _symbols_to_sequence_en(['@spn'])
        sequence += start
        while len(text):
            m = _curly_re.match(text)
            if not m:
                clean_text = _clean_text(text, cleaner_names)
                if dictionary is not None:
                    clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
                    for i in range(len(clean_text)):
                        t = clean_text[i]
                        if t.startswith("{"):
                            sequence += _arpabet_to_sequence_en(t[1:-1])
                        else:
                            sequence += _symbols_to_sequence_en(t)
                        sequence += space
                else:
                    sequence += _symbols_to_sequence_en(clean_text)
                break
            sequence += _symbols_to_sequence_en(_clean_text(m.group(1), cleaner_names))
            sequence += _arpabet_to_sequence_en(m.group(2))
            text = m.group(3)
        if dictionary is not None:
            sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
        sequence += end
    if language == 'ch':
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += _symbols_to_sequence_ch(_clean_text(text, cleaner_names))
                break
            sequence += _symbols_to_sequence_ch(_clean_text(m.group(1), cleaner_names))
            sequence += _arpabet_to_sequence_ch(m.group(2))
            text = m.group(3)
    return sequence


def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence_en(symbols):
    return [_symbol_to_id_en[s] for s in symbols if _should_keep_symbol_en(s)]


def _arpabet_to_sequence_en(text):
    return _symbols_to_sequence_en(['@' + s for s in text.split()])


def _should_keep_symbol_en(s):
    return s in _symbol_to_id_en and s != '_' and s != '~'


def _symbols_to_sequence_ch(symbols):
    return [_symbol_to_id_ch[s] for s in symbols if _should_keep_symbol_ch(s)]


def _arpabet_to_sequence_ch(text):
    return _symbols_to_sequence_ch(['@' + s for s in text.split()])


def _should_keep_symbol_ch(s):
    return s in _symbol_to_id_ch and s != '_' and s != '~'

def _symbols_to_sequence_fr(symbols):
    return [_symbol_to_id_fr[s] for s in symbols if _should_keep_symbol_fr(s)]


def _arpabet_to_sequence_fr(text):
    return _symbols_to_sequence_fr(['@' + s for s in text.split()])


def _should_keep_symbol_fr(s):
    return s in _symbol_to_id_fr and s != '_' and s != '~'