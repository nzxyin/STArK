_pad = ["_"]

_word_boundaries = ['|']
_stress = ['*']
_silences = ['<sp>', '<eos>', '<exclm>', '<quest>']

_phonemes = ['n', 't', 'ɪ', 'd', 's', 'ə', 'æ', 'l', 'ð', 'ɹ', 'ʌ', 'm', 'k', 'z', 'ɛ', 'w', 'h', 'p', 'v', 'iː', 'f', 'b', 'aɪ', 'ɚ', 'eɪ', 'oʊ', 'ɐ', 'i', 'ŋ', 'uː', 'ᵻ', 'ɑː', 
             'ɡ', 'ɜː', 'ʃ', 'ɾ', 'j', 'ʊ', 'aʊ', 'əl', 'tʃ', 'ɔː', 'θ', 'dʒ', 'ɔːɹ', 'ɑːɹ', 'ɔ', 'ɛɹ', 'oːɹ', 'ɪɹ', 'iə', 'ʊɹ', 'ɔɪ', 'oː', 'aɪɚ', 'ʒ', 'n̩', 'ʔ', 'aɪə',]

symbols = _pad + _word_boundaries + _stress + _phonemes + _silences

_symbol_to_id = {symbols[i]: i for i in range(len(symbols))}

_id_to_symbol = {i: symbols[i] for i in range(len(symbols))}

def get_id(symbol: str) -> int:
    return _symbol_to_id[symbol]

def get_symbol(id: int) -> str:
    return _id_to_symbol[id]