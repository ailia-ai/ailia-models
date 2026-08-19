import sys
import time
import re
from logging import getLogger

import numpy as np

import re
import os
import unicodedata
from builtins import str as unicode
from expand import normalize_numbers
import numpy as np
import ailia

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

ENCODER_WEIGHT_PATH = "g2p_encoder.onnx"
ENCODER_MODEL_PATH = "g2p_encoder.onnx.prototxt"
DECODER_WEIGHT_PATH = "g2p_decoder.onnx"
DECODER_MODEL_PATH = "g2p_decoder.onnx.prototxt"

CMUDICT_PATH = "cmudict"
HOMOGRAPHS_PATH = "homographs.en"
TAGGER_PATH = "averaged_perceptron_tagger.pickle"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/g2p_en/"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("g2p_en: Grapheme-to-Phoneme", None, None)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="I'm an activationist.",
    help="Input text.",
)
parser.add_argument("--verify", action="store_true", help="verify mode.")
parser.add_argument("--onnx", action="store_true", help="use onnx runtime.")
args = update_parser(parser, check_input_type=False)


# ======================
# Dictionary
# ======================

dirname = os.path.dirname(__file__)

def construct_homograph_dictionary():
    f = os.path.join(dirname,'homographs.en')
    homograph2features = dict()
    for line in open(f, 'r', encoding='utf-8').read().splitlines():
        if line.startswith("#"): continue # comment
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features

def construct_cmu_dictionary():
    f = os.path.join(dirname,'cmudict')
    cmudict = dict()
    for line in open(f, 'r', encoding='utf-8').read().splitlines():
        if line.startswith("#"): continue # comment
        lists = line.strip().split(" ")
        headword = lists[0]
        pron = lists[2:]
        key = headword.lower()
        if not (key in cmudict):
            cmudict[key] = pron
    return cmudict

class G2p(object):
    def __init__(self):
        super().__init__()

        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.homograph2features = construct_homograph_dictionary()
        self.cmudict = construct_cmu_dictionary()

    def tokenize(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        return np.array(x)

    def predict(self, word, encoder, decoder):
        x = self.tokenize(word)

        h = np.zeros((1, 256), dtype=np.float32)

        if args.onnx:
            for i in range(x.shape[0]):
                h = encoder.run(
                    None,
                    {
                        'x': np.array([x[i]]),
                        'h_in': h
                    },
                )[0]
        else:
            for i in range(x.shape[0]):
                h = encoder.run([np.array([x[i]]), h])[0]

        preds = []
        pred = 2    # initial symbol
        for i in range(20):
            pred = np.array([pred])
            if args.onnx:
                logits, h = decoder.run(
                    None,
                    {
                        'pred': pred,
                        'h_in': h,
                    },
                )
            else:
                logits, h = decoder.run([pred, h])
            pred = np.argmax(logits)
            if pred == 3: break  # 3: </s>
            preds.append(pred)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text, encoder, decoder):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # word tokenize
        print(text)
        text2 = text
        text2 = text2.replace(".", " . ") # 句読点を単独トークンにする
        text2 = text2.replace(",", " , ")
        text2 = text2.replace("!", " ! ")
        text2 = text2.replace("?", " ? ")
        words = text2.split()

        # classify 
        from averaged_perceptron import tag
        tokens = tag(words)

        # steps
        prons = []
        for word, pos in tokens:
            if args.verify:
                pron = self.predict(word, encoder, decoder)
            
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmudict:  # lookup CMU dict
                pron = self.cmudict[word]
            else: # predict for oov
                if args.benchmark:
                    logger.info("BENCHMARK mode")
                    total_time_estimation = 0
                    for i in range(args.benchmark_count):
                        start = int(round(time.time() * 1000))
                        pron = self.predict(word, encoder, decoder)
                        end = int(round(time.time() * 1000))
                        estimation_time = end - start

                        # Logging
                        logger.info(f"\tailia processing estimation time {estimation_time} ms")
                        if i != 0:
                            total_time_estimation = total_time_estimation + estimation_time

                    logger.info(
                        f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
                    )
                else:
                    pron = self.predict(word, encoder, decoder)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

def recognize_from_text(encoder, decoder):
    g2p = G2p()
    out = g2p(args.input, encoder, decoder)

    logger.info("Input : " + str(args.input))
    logger.info("Output : " + str(out))
    logger.info("Script finished successfully.")


def verify(encoder, decoder):
    texts = ["I have $250 in my pocket.", # number -> spell-out
             "popular pets, e.g. cats and dogs", # e.g. -> for example
             "I refuse to collect the refuse around here.", # homograph
             "I'm an activationist."] # newly coined word
    references = [['AY1', ' ', 'HH', 'AE1', 'V', ' ', 'T', 'UW1', ' ', 'HH', 'AH1', 'N', 'D', 'R', 'AH0', 'D', ' ', 'F', 'IH1', 'F', 'T', 'IY0', ' ', 'D', 'AA1', 'L', 'ER0', 'Z', ' ', 'IH0', 'N', ' ', 'M', 'AY1', ' ', 'P', 'AA1', 'K', 'AH0', 'T', ' ', '.'],
['P', 'AA1', 'P', 'Y', 'AH0', 'L', 'ER0', ' ', 'P', 'EH1', 'T', 'S', ' ', ',', ' ', 'F', 'AO1', 'R', ' ', 'IH0', 'G', 'Z', 'AE1', 'M', 'P', 'AH0', 'L', ' ', 'K', 'AE1', 'T', 'S', ' ', 'AH0', 'N', 'D', ' ', 'D', 'AA1', 'G', 'Z'],
['AY1', ' ', 'R', 'IH0', 'F', 'Y', 'UW1', 'Z', ' ', 'T', 'UW1', ' ', 'K', 'AH0', 'L', 'EH1', 'K', 'T', ' ', 'DH', 'AH0', ' ', 'R', 'EH1', 'F', 'Y', 'UW2', 'Z', ' ', 'ER0', 'AW1', 'N', 'D', ' ', 'HH', 'IY1', 'R', ' ', '.'],
['AY1', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'IH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T', ' ', '.']]
    g2p = G2p()
    for text,reference in zip(texts, references):
        out = g2p(text, encoder, decoder)
        if out != reference:
            print("Verify Error")
            print(out)
            print(reference)
            exit()
    print("Verify Success")


def main():
    # model files check and download
    check_and_download_models(ENCODER_WEIGHT_PATH, ENCODER_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DECODER_WEIGHT_PATH, DECODER_MODEL_PATH, REMOTE_PATH)
    check_and_download_file(CMUDICT_PATH, REMOTE_PATH)
    check_and_download_file(HOMOGRAPHS_PATH, REMOTE_PATH)
    check_and_download_file(TAGGER_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if args.onnx:
        import onnxruntime
        providers = ['CPUExecutionProvider']
        encoder = onnxruntime.InferenceSession(ENCODER_WEIGHT_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(DECODER_WEIGHT_PATH, providers=providers)
    else:
        encoder = ailia.Net(ENCODER_MODEL_PATH, ENCODER_WEIGHT_PATH, env_id=env_id)
        decoder = ailia.Net(DECODER_MODEL_PATH, DECODER_WEIGHT_PATH, env_id=env_id)

    if args.verify:
        verify(encoder, decoder)
    else:
        recognize_from_text(encoder, decoder)


if __name__ == "__main__":
    main()
