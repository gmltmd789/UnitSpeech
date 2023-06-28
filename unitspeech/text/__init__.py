""" from https://github.com/keithito/tacotron """
from unitspeech.text import cleaners
from unitspeech.text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def phonemize(text, global_phonemizer):
    text = cleaners.convert_to_ascii(text)
    text = cleaners.lowercase(text)
    text = cleaners.expand_abbreviations(text)
    phonemes = global_phonemizer.phonemize([text], strip=True)[0]
    phonemes = cleaners.collapse_whitespace(phonemes)
    return phonemes


def cleaned_text_to_sequence(cleaned_text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence
