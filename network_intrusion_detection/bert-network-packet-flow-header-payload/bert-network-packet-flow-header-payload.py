import sys
import os
import time
from logging import getLogger

import numpy as np
from scapy.all import Ether, CookedLinux

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models
from math_utils import softmax


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "model.onnx"
MODEL_PATH = "model.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert-network-packet-flow-header-payload/"

LABELS = [
    "Analysis",
    "Backdoor",
    "Bot",
    "DDoS",
    "DoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS SlowHTTPTest",
    "DoS Slowloris",
    "Exploits",
    "FTP Patator",
    "Fuzzers",
    "Generic",
    "Heartbleed",
    "Infiltration",
    "Normal",
    "Port Scan",
    "Reconnaissance",
    "SSH Patator",
    "Shellcode",
    "Web Attack - Brute Force",
    "Web Attack - SQL Injection",
    "Web Attack - XSS",
    "Worms",
]

PACEKT_HEX_PATH = "input_hex.txt"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "bert-network-packet-flow-header-payload",
    PACEKT_HEX_PATH,
    None,
)
parser.add_argument("--hex", type=str, default=None, help="Input-HEX data.")
parser.add_argument("--ip", action="store_true", help="Use IP layer as payload.")
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================


def preprocess(packet_hex):
    packet_bytes = bytes.fromhex(packet_hex)
    packet = Ether(packet_bytes)
    if packet.firstlayer().name != "Ethernet":
        packet = CookedLinux(packet_bytes)
        if packet.firstlayer().name != "cooked linux":
            raise ValueError(
                f"{packet.firstlayer().name} frame not implemented. Ethernet and Cooked Linux are only supported."
            )

    if "IP" not in packet or "TCP" not in packet:
        raise ValueError("Only TCP/IP packets are supported.")

    forward_packets = 0
    backward_packets = 0
    bytes_transfered = len(packet_bytes)

    # Extract relevant information for feature creation.
    src_ip = packet["IP"].src
    dst_ip = packet["IP"].dst
    ip_length = len(packet["IP"])
    ip_ttl = packet["IP"].ttl
    ip_tos = packet["IP"].tos
    src_port = packet["TCP"].sport
    dst_port = packet["TCP"].dport
    tcp_data_offset = packet["TCP"].dataofs
    tcp_flags = packet["TCP"].flags

    # Process payload content and create a feature string.
    payload_bytes = bytes(packet["IP"].payload if args.ip else packet["TCP"].payload)
    payload_length = len(payload_bytes)
    payload_decimal = [str(byte) for byte in payload_bytes]

    final_data = [
        forward_packets,
        backward_packets,
        bytes_transfered,
        -1,
        src_port,
        dst_port,
        ip_length,
        payload_length,
        ip_ttl,
        ip_tos,
        tcp_data_offset,
        int(tcp_flags),
        -1,
    ] + payload_decimal

    final_data = " ".join(str(s) for s in final_data)
    return final_data


def predict(models, packet_hex):
    final_format = preprocess(packet_hex)

    tokenizer = models["tokenizer"]
    model_inputs = tokenizer(final_format[:1024], return_tensors="np")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    net = models["net"]

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, attention_mask])
    else:
        output = net.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
    logits = output[0]

    scores = softmax(logits[0])
    idx = np.argsort(-scores)
    labels = np.array(LABELS)[idx]
    scores = scores[idx]

    return (labels, scores)


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

    top_k = 3
    labels, socres = output
    for label, score in list(zip(labels, socres))[:top_k]:
        print(f"{label} : {score*100:.3f}")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=args.memory_mode)
        pass
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("./tokenizer/")

    models = {
        "tokenizer": tokenizer,
        "net": net,
    }

    recognize_from_packet(models)


if __name__ == "__main__":
    main()
