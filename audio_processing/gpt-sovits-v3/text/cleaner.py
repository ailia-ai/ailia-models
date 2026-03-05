from . import symbols2 as symbols_v2


def clean_text(text, language):
    symbols = symbols_v2.symbols
    language_module_map = {
        "zh": "chinese2",
        "ja": "japanese",
        "en": "english",
        # "ko": "korean",
        # "yue": "cantonese",
    }

    if language not in language_module_map:
        language = "en"
        text = " "
    language_module = __import__(
        "text." + language_module_map[language],
        fromlist=[language_module_map[language]],
    )
    norm_text = language_module.text_normalize(text)
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ["UNK" if ph not in symbols else ph for ph in phones]

    return phones, word2ph, norm_text
