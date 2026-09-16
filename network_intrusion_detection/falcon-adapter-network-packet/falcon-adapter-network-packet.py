import sys
import os
import time
import re
import itertools
import joblib
import warnings
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scapy.all import Ether, CookedLinux

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models, check_and_download_file

nltk.download("stopwords")
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SIMILAR_PATH = "similar.onnx"
MODEL_SIMILAR_PATH = "similar.onnx.prototxt"
REMOTE_PATH = (
    "https://storage.googleapis.com/ailia-models/falcon-adapter-network-packet/"
)

PACEKT_HEX_PATH = "input_hex.txt"

SCALER_PKL = "SCALER.pkl"
CLUSTER_CENTERS = "KMEANS-CLUSTER-CENTERS.npy"
TAGS_NAMES = "TAGS-NAMES-EMBEDDINGS.npy"

BERT_VECTORS_ALL = "bert_vectors_all.npy"
BERT_WORDS_ALL = "bert_words_all.npy"

CORPUS_PATH = "CORPUS.txt"


max_document_length = 375
max_heading_length = 10
max_ngram = 10
exclude_stopwords = ["dos"]
stop_words = [word for word in stopwords.words() if word not in exclude_stopwords]
doc_regex = "[\([][0-9]+[\])]|[”“‘’‛‟]|\d+\s"
punctuations = """!"#$%&'()*+,-./:—;<=>−?–@[\]^_`{|}~"""
punctuations_continuity_exclude = """—-–,−"""


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "falcon-adapter-network-packet",
    PACEKT_HEX_PATH,
    None,
    fp16_support=False
)
parser.add_argument("--hex", type=str, default=None, help="Input-HEX data.")
parser.add_argument(
    "--save_vector", action="store_true", help="Save the generated vector."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


@dataclass
class _SimilarWords:
    bert_words = []
    bert_vectors = []
    bert_documents = []
    continous_words = []
    temporary_ngram_words = []
    count_vectorizer_words = []
    cv_counts = []
    cv_words = []

    bert_words_ngram = [[] for _ in range(max_ngram)]
    bert_vectors_ngram = [[] for _ in range(max_ngram)]
    bert_documents_ngram = [[] for _ in range(max_ngram)]
    bert_words_all = []
    bert_vectors_all = []


similar = _SimilarWords()


def process_txt_dataset(path):
    document_list = []
    with open(path, encoding="utf8") as file:
        for line in tqdm(
            file.readlines(),
            unit=" paragraphs",
            desc="Extracting",
            postfix="Data from Dataset",
        ):
            line_text = line.strip()
            line_text = re.sub(doc_regex, "", line_text)
            line_length = len(line_text.split())
            if 0 < line_length <= max_heading_length:
                if (
                    len(document_list) != 0
                    and len(document_list[-1].split()) <= max_heading_length
                ):
                    document_list[-1] = line_text + "."
                else:
                    document_list.append(line_text + ".")
            elif line_length > max_heading_length:
                if (
                    len(document_list) != 0
                    and len(document_list[-1].split()) + line_length
                    <= max_document_length
                ):
                    document_list[-1] += " " + line_text
                else:
                    document_list = process_dataset_long_paragraph(
                        document_list,
                        line_text,
                        len(document_list[-1].split()) + line_length,
                    )

    return document_list


def process_dataset_long_paragraph(document_list, sentence, sentence_length):
    if sentence_length > max_document_length:
        for i in range(2, sentence_length):
            div = sentence_length / i
            if div < max_document_length:
                break
        temp_sent = ""
        sm_sent = sent_tokenize(sentence)

        for sent in sm_sent:
            if len(temp_sent.split() + sent.split()) > div:
                if len(document_list[-1].split()) <= max_heading_length:
                    document_list[-1] += " " + temp_sent
                else:
                    document_list.append(temp_sent)
                temp_sent = ""
            temp_sent = temp_sent + sent

        if len(document_list[-1].split() + temp_sent.split()) < max_document_length:
            document_list[-1] += " " + temp_sent
        else:
            document_list.append(temp_sent)
    else:
        document_list.append(sentence)

    return document_list


def tokenize_and_embeddings(models, document_list):
    tokenizer = models["tokenizer"]
    net = models["similar"]

    continous_index = 0
    document_index = 0
    for document in tqdm(
        document_list, unit=" documents", desc="Processing", postfix="Word Embeddings"
    ):
        # for document in document_list:
        tokens = tokenizer(document, truncation=True, max_length=512, return_tensors="np")
        words = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist())
        word_ids = tokens.word_ids()

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # feedforward
        if not args.onnx:
            output = net.predict([input_ids, attention_mask])
        else:
            output = net.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )
        last_hidden_state = output[0]
        vectors = last_hidden_state[0]

        word_list = []
        vector_list = []
        continous_words = []
        word_index = -1
        for i in range(len(words)):
            if word_ids[i] is None or words[i] in punctuations:
                if words[i] in punctuations_continuity_exclude:
                    pass
                else:
                    continous_index = continous_index + 1
                continue
            if word_ids[i] > word_index:
                if len(word_list) != 0 and word_list[-1].lower() in stop_words:
                    word_list.pop()
                    vector_list.pop()
                    continous_words.pop()
                    continous_index = continous_index + 1
                word_list.append(words[i])
                vector_list.append(vectors[i])
                continous_words.append(continous_index)
                word_index = word_ids[i]
            elif word_ids[i] == word_index:
                sub_word = words[i].replace("##", "")
                word_list[-1] = word_list[-1] + sub_word
                vector_list[-1] = vector_list[-1] + vectors[i]
                if word_ids[i + 1] != word_ids[i]:
                    vector_list[-1] = vector_list[-1] / word_ids.count(word_index)

        yield word_list, vector_list, [document_index] * len(word_list), continous_words
        document_index += 1


def generate_n_grams(i, words, vectors, document, continous, n=1):
    if i > n - 1 and n < max_ngram and continous[i] == continous[i - n]:
        temp_word = ""
        temp_vector = np.zeros([len(vectors[i])])
        for j in range(n, -1, -1):
            temp_word = temp_word + " " + words[i - j]
            temp_vector = temp_vector + vectors[i - j]
        similar.temporary_ngram_words.append(temp_word.strip())
        similar.bert_words_ngram[n].append(temp_word.strip())
        similar.bert_vectors_ngram[n].append(temp_vector / (n + 1))
        similar.bert_documents_ngram[n].append(document[i])
        generate_n_grams(i, words, vectors, document, continous, n=n + 1)

    return


def custom_analyzer(words):
    final_list = []
    for word in words:
        final_list.append(word)
        lemmatized_word = " ".join(
            [
                custom_analyzer.lemmatizer.lemmatize(token.lower())
                for token in word.split()
            ]
        )
        if word != lemmatized_word:
            final_list.append(lemmatized_word)
    return final_list


custom_analyzer.lemmatizer = WordNetLemmatizer()
count_vectorizer = CountVectorizer(analyzer=custom_analyzer)


def load_dataset(models, dataset_path):
    document_list = process_txt_dataset(dataset_path)
    for words, vectors, document, continous in tokenize_and_embeddings(
        models, document_list
    ):
        temporary_ngram_words = []
        for i in range(len(words)):
            generate_n_grams(i, words, vectors, document, continous)
        similar.bert_words.extend(words)
        similar.bert_vectors.extend(vectors)
        similar.bert_documents.extend(document)
        similar.continous_words.extend(continous)
        similar.count_vectorizer_words.append(words + temporary_ngram_words)

    similar.bert_words_ngram[0] = similar.bert_words
    similar.bert_vectors_ngram[0] = similar.bert_vectors
    similar.bert_documents_ngram[0] = similar.bert_documents
    similar.cv_counts = count_vectorizer.fit_transform(similar.count_vectorizer_words)
    similar.cv_words = count_vectorizer.get_feature_names_out()
    similar.bert_words_all = np.array(
        list(itertools.chain.from_iterable(similar.bert_words_ngram))
    )
    similar.bert_vectors_all = np.array(
        list(itertools.chain.from_iterable(similar.bert_vectors_ngram))
    )

    scaler = models["scaler"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        similar.bert_vectors_all = scaler.transform(similar.bert_vectors_all)
        for i in tqdm(
            range(max_ngram), desc="Generating", postfix="N-gram Words and Embeddings"
        ):
            similar.bert_vectors_ngram[i] = scaler.transform(
                similar.bert_vectors_ngram[i]
            )


def find_similar_words(
    input_embedding,
    pos_to_exclude=[],
    max_output_words=10,
    output_filter_factor=0.5,
):
    input_context_words = []

    a = []
    for i in range(0, len(similar.bert_vectors_all), 100000):
        a.append(
            cosine_similarity(
                similar.bert_vectors_all[i : i + 100000, :], [input_embedding]
            ).flatten()
        )
    cosine_sim = np.concatenate(a)
    cosine_words = similar.bert_words_all

    output_dict = {}
    sorted_list = np.flip(np.argsort(cosine_sim))
    lemmatized_words = {
        custom_analyzer.lemmatizer.lemmatize(token.lower())
        for word in input_context_words
        for token in word.split()
    }

    for i in range(len(cosine_words)):
        stop = 0
        pop_list = []
        original_word = cosine_words[sorted_list[i]]
        pos_tags = [pos[1] for pos in nltk.pos_tag(original_word.split())]
        lemmatized_word = {
            custom_analyzer.lemmatizer.lemmatize(token.lower())
            for token in original_word.split()
        }
        if len(
            lemmatized_words.intersection(lemmatized_word)
        ) > output_filter_factor * len(original_word.split()):
            continue
        if any(pos in pos_tags for pos in pos_to_exclude):
            continue
        if original_word not in output_dict.keys():
            for word in output_dict.keys():
                if original_word in word:
                    stop = 1
                    break
                elif word in original_word:
                    pop_list.append(word)
                    stop = 0
            if stop == 0:
                pop = [output_dict.pop(key) for key in pop_list]
                output_dict[original_word] = cosine_sim[sorted_list[i]]
                if len(output_dict.keys()) == max_output_words:
                    break

    return output_dict, input_embedding


# ======================
# Main functions
# ======================


def predict(models, packet_hex):
    packet_bytes = bytes.fromhex(packet_hex)
    packet = Ether(packet_bytes)
    if packet.firstlayer().name != "Ethernet":
        packet = CookedLinux(packet_bytes)
        if packet.firstlayer().name != "cooked linux":
            raise ValueError(
                f"{packet.firstlayer().name} frame not implemented. Ethernet and Cooked Linux are only supported."
            )

    forward_packets_per_second = 0
    backward_packets_per_second = 4
    bytes_transferred_per_second = 5493

    source_port = packet["TCP"].sport
    destination_port = packet["TCP"].dport
    IP_len = packet["IP"].len
    IP_ttl = packet["IP"].ttl
    IP_tos = f"0x{str(packet['IP'].tos)}"
    tos_map = {
        "0x0": 0,
        "0x10": 1,
        "0x18": 2,
        "0x2": 3,
        "0x20": 4,
        "0x28": 5,
        "0x34": 6,
        "0x4": 7,
        "0x40": 8,
        "0x48": 9,
        "0x60": 10,
        "0x68": 11,
        "0x8": 12,
        "0x88": 13,
    }
    IP_tos = tos_map.get(IP_tos, 14)
    TCP_dataofs = packet["TCP"].dataofs
    TCP_flags = str(packet["TCP"].flags)
    flags_map = {"A": 0, "FA": 1, "FPA": 2, "PA": 3}
    TCP_flags = flags_map.get(TCP_flags, 4)
    if packet.haslayer("Raw"):
        payload_hex = packet.load.hex()
        payload_len = len(payload_hex) // 2
    else:
        raise ValueError(
            "Network Packet does not contain a payload. This model is trained with a payload."
        )
    payload_hex = packet.load.hex()
    payload_hex = payload_hex

    payload_len = len(payload_hex) // 2
    payload = [int(payload_hex[i : i + 2], 16) for i in range(0, len(payload_hex), 2)]

    final_format = [
        forward_packets_per_second,
        backward_packets_per_second,
        bytes_transferred_per_second,
        -1,
        source_port,
        destination_port,
        IP_len,
        payload_len,
        IP_ttl,
        IP_tos,
        TCP_dataofs,
        TCP_flags,
        -1,
    ]
    final_format = final_format + payload[:500]
    final_format = [str(i) for i in final_format]
    final_format = " ".join(final_format)

    tokenizer = models["tokenizer"]
    net = models["similar"]
    tokens = tokenizer(final_format, truncation=True, max_length=512, return_tensors="np")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, attention_mask])
    else:
        output = net.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
    last_hidden_state = output[0]

    embedding = np.mean(last_hidden_state[:, 1:-1, :], axis=1)

    scaler = models["scaler"]
    df = pd.DataFrame(embedding[0].reshape(1, -1), columns=[str(i) for i in range(768)])
    embedding = scaler.transform(df)[0]

    cluster_centers = models["cluster_centers"]
    tag_names = models["tag_names"]

    euclidean_distance = euclidean_distances(cluster_centers, [embedding])
    data = {i: euclidean_distance[i][0] for i in range(len(euclidean_distance))}
    df = pd.DataFrame(list(data.items()), columns=["Cluster", "Euclidean Distance"])
    df.sort_values(by="Euclidean Distance", inplace=True)

    cluster, distance, embedding = (
        int(df.iloc[0]["Cluster"]),
        df.iloc[0]["Euclidean Distance"],
        embedding,
    )

    total_tags = 10
    output_filter_factor = 1
    tags, emb = find_similar_words(
        input_embedding=(tag_names[cluster] + (embedding - cluster_centers[cluster])),
        max_output_words=total_tags,
        output_filter_factor=output_filter_factor,
    )

    return (tags, emb)


def recognize_from_packet(models):
    packet_hex = args.hex
    if packet_hex:
        args.input[0] = packet_hex

    # input audio loop
    for packet_path in args.input:
        # prepare input data
        if os.path.isfile(packet_path):
            logger.info(packet_path)
            with open(packet_path, "r") as f:
                packet_hex = f.read()

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, packet_hex)
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
            output = predict(models, packet_hex)

    tags = output[0]
    tags = sorted([(t, p) for t, p in tags.items()], key=lambda x: x[1], reverse=True)
    print("--- tags ---")
    for t, p in tags:
        print(f"{t} ({p:.4f})")
    print("------------")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_SIMILAR_PATH, MODEL_SIMILAR_PATH, REMOTE_PATH)
    check_and_download_file(CORPUS_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_similar = ailia.Net(MODEL_SIMILAR_PATH, WEIGHT_SIMILAR_PATH, env_id=env_id, memory_mode=args.memory_mode)
    else:
        import onnxruntime
        net_similar = onnxruntime.InferenceSession(WEIGHT_SIMILAR_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("./tokenizer/")
    scaler = joblib.load(SCALER_PKL)
    cluster_centers = np.load(CLUSTER_CENTERS)
    tag_names = np.load(TAGS_NAMES)

    models = {
        "tokenizer": tokenizer,
        "similar": net_similar,
        "scaler": scaler,
        "cluster_centers": cluster_centers,
        "tag_names": tag_names,
    }

    if os.path.exists(BERT_VECTORS_ALL) and os.path.exists(BERT_WORDS_ALL):
        logger.info("Loading vector...")
        similar.bert_vectors_all = np.load(BERT_VECTORS_ALL)
        similar.bert_words_all = np.load(BERT_WORDS_ALL)
    else:
        load_dataset(models, CORPUS_PATH)
        if args.save_vector:
            logger.info("Saving vector...")
            np.save(BERT_VECTORS_ALL, similar.bert_vectors_all)
            np.save(BERT_WORDS_ALL, similar.bert_words_all)

    recognize_from_packet(models)


if __name__ == "__main__":
    main()
