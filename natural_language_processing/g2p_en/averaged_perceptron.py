import pickle
from collections import defaultdict

# python 2 系の __builtin__.set を持った pickle をロードする互換 unpickler
class CompatUnpickler(pickle.Unpickler):
    ALLOWED = [
        ("__builtin__", "set"),
    ]

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"forbidden global: {module}.{name}")

EXPORT_TO_TEXT = False
IMPORT_FROM_TEXT = False
UNIT_TEST = False

class AveragedPerceptron:
    def __init__(self, weights=None):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = weights if weights else {}
        self.classes = set()

    def predict(self, features):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight

        # Do a secondary alphabetic sort, for stability
        best_label = max(self.classes, key=lambda label: (scores[label], label))

        return best_label

model = AveragedPerceptron()

def export_to_text():
    with open("averaged_perceptron_tagger.pickle", "rb") as f:
        weights, tagdict, classes = CompatUnpickler(f).load()
    f = open("averaged_perceptron_tagger_weights.txt", "w")
    for feat in weights.keys():
        feat_weights = weights[feat]
        for label, weight in feat_weights.items():
            f.write(feat + "\n" + label + "\n" + str(weight) + "\n")
    f.close()

    f = open("averaged_perceptron_tagger_tagdict.txt", "w")
    for tag in tagdict.keys():
        f.write(tag + "\n" + tagdict[tag] + "\n")
    f.close()

    f = open("averaged_perceptron_tagger_classes.txt", "w")
    for cls in classes:
        f.write(cls + "\n")
    f.close()

def import_from_text():
    f = open("averaged_perceptron_tagger_weights.txt", "r")
    weights = {}
    lines = f.read().split('\n')[:-1]
    i = 0
    while i < len(lines):
        feat = lines[i+0]
        label = lines[i+1]
        weight = lines[i+2]
        i = i + 3
        if not (feat in weights):
            weights[feat] = {}
        weights[feat][label] = float(weight)
    f.close()

    f = open("averaged_perceptron_tagger_tagdict.txt", "r")
    tagdict = {}
    lines = f.read().split('\n')[:-1]
    i = 0
    while i < len(lines):
        tag = lines[i+0]
        v = lines[i+1]
        i = i + 2
        tagdict[tag] = v
    f.close()
        
    f = open("averaged_perceptron_tagger_classes.txt", "r")
    classes = f.read().split('\n')[:-1]
    f.close()
    return weights, tagdict, classes

if EXPORT_TO_TEXT:
    export_to_text()
if IMPORT_FROM_TEXT:
    model.weights, tagdict, model.classes = import_from_text()
else:
    with open("averaged_perceptron_tagger.pickle", "rb") as f:
        model.weights, tagdict, model.classes = CompatUnpickler(f).load()

START = ["-START-", "-START2-"]
END = ["-END-", "-END2-"]

def _get_features(i, word, context, prev, prev2):
    """Map tokens into a feature representation, implemented as a
    {hashable: int} dict. If the features change, a new model must be
    trained.
    """

    def add(name, *args):
        features[" ".join((name,) + tuple(args))] += 1

    i += len(START)
    features = defaultdict(int)
    # It's useful to have a constant feature, which acts sort of like a prior
    add("bias")
    add("i suffix", word[-3:])
    add("i pref1", word[0] if word else "")
    add("i-1 tag", prev)
    add("i-2 tag", prev2)
    add("i tag+i-2 tag", prev, prev2)
    add("i word", context[i])
    add("i-1 tag+i word", prev, context[i])
    add("i-1 word", context[i - 1])
    add("i-1 suffix", context[i - 1][-3:])
    add("i-2 word", context[i - 2])
    add("i+1 word", context[i + 1])
    add("i+1 suffix", context[i + 1][-3:])
    add("i+2 word", context[i + 2])
    return features

def normalize(word):
    """
    Normalization used in pre-processing.
    - All words are lower cased
    - Groups of digits of length 4 are represented as !YEAR;
    - Other digits are represented as !DIGITS

    :rtype: str
    """
    if "-" in word and word[0] != "-":
        return "!HYPHEN"
    if word.isdigit() and len(word) == 4:
        return "!YEAR"
    if word and word[0].isdigit():
        return "!DIGITS"
    return word.lower()

def tag(tokens):
    """
    Tag tokenized sentences.
    :params tokens: list of word
    :type tokens: list(str)
    """
    prev, prev2 = START
    output = []

    context = START + [normalize(w) for w in tokens] + END
    for i, word in enumerate(tokens):
        tag = tagdict.get(word)
        if not tag:
            features = _get_features(i, word, context, prev, prev2)
            tag = model.predict(features)
        output.append((word, tag))

        prev2 = prev
        prev = tag

    return output

if UNIT_TEST:
    words = ["i'm", 'an', 'activationist', '.']
    output = tag(words)
    print(output)
    #[("i'm", 'VB'), ('an', 'DT'), ('activationist', 'NN'), ('.', '.')]

