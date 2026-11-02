import multiprocessing as mp
import queue
import sys
import threading
import time
from logging import getLogger

import numpy as np

logger = getLogger(__name__)

M1_SAMPLE_RATE = 48000


def capture_microphone(que, ready, pause, fin, sample_rate, sc=False, speaker=False):
    if sc:
        import soundcard as sc
    else:
        import librosa
        import pyaudio

    if sc:
        SAMPLE_RATE = sample_rate
    else:
        p = pyaudio.PyAudio()
        # M1 macだとpyaudioが48000Hzしか取得できないので、指定のサンプルレートで開けない場合は48000Hzで取得して変換する
        try:
            device = p.get_default_input_device_info()
            p.is_format_supported(
                sample_rate,
                input_device=device["index"],
                input_channels=1,
                input_format=pyaudio.paInt16,
            )
            SAMPLE_RATE = sample_rate
        except (ValueError, OSError):
            SAMPLE_RATE = M1_SAMPLE_RATE
    THRES_SPEECH_POW = 0.001
    THRES_SILENCE_POW = 0.0001
    INTERVAL = SAMPLE_RATE * 3
    INTERVAL_MIN = SAMPLE_RATE * 1.5
    BUFFER_MAX = SAMPLE_RATE * 10

    def send(audio, n):
        if INTERVAL_MIN < n:
            audio = audio[:n]
            if sc is False and SAMPLE_RATE != sample_rate:
                audio = librosa.resample(
                    audio, orig_sr=SAMPLE_RATE, target_sr=sample_rate
                )
            try:
                que.put_nowait(audio)
            except queue.Full:
                # 受信側の処理が追いつかない場合は、キャプチャを止めずにこの区間を捨てる
                logger.warning(
                    "microphone input queue is full, the captured audio is dropped."
                )

    def read(src):
        ready.set()

        v = np.ones(100) / 100
        buf = np.array([], dtype=np.float32)
        while not fin.is_set():
            if pause.is_set():
                buf = buf[:0]
                time.sleep(0.1)
                continue

            if sc:
                audio = src.record(INTERVAL)
            else:
                audio = (
                    np.frombuffer(
                        src.read(INTERVAL, exception_on_overflow=False), dtype=np.int16
                    )
                    / 32768.0
                )

            audio = audio.reshape(-1)
            square = audio**2
            if np.max(square) >= THRES_SPEECH_POW:
                sys.stdout.write(".")
                sys.stdout.flush()

                # 平準化
                conv = np.convolve(square, v, "valid")
                conv = np.pad(conv, (0, len(v) - 1), mode="edge")
                # 0.1s刻みで0.5s区間をチェック
                s = SAMPLE_RATE // 10
                x = [
                    (
                        min(i + s * 5, INTERVAL),
                        np.any(conv[i : i + s * 5] >= THRES_SILENCE_POW),
                    )
                    for i in range(0, INTERVAL - 5 * s + 1, s)
                ]

                # Speech section
                speech = [a[0] for a in x if a[1]]
                if speech:
                    if len(buf) == 0 and 1 < len(speech):
                        i = max(speech[0] - 3 * s, 0)
                        audio = audio[i:]
                    i = speech[-1]
                    audio = audio[:i]
                else:
                    i = 0

                buf = np.concatenate([buf, audio])
                if i < INTERVAL:
                    send(buf, len(buf))
                    buf = buf[:0]
                elif BUFFER_MAX < len(buf):
                    i = np.argmin(buf[::-1])
                    i = len(buf) - i
                    if 0 < i:
                        send(buf, i)
                        buf = buf[i:]
                    else:
                        send(buf, len(buf))
                        buf = buf[:0]
            elif 0 < len(buf):
                send(buf, len(buf))
                buf = buf[:0]

    try:
        # start recording
        if sc:
            mic_id = (
                str(sc.default_speaker().name)
                if speaker
                else str(sc.default_microphone().name)
            )
            with sc.get_microphone(id=mic_id, include_loopback=speaker).recorder(
                samplerate=SAMPLE_RATE, channels=1
            ) as mic:
                read(mic)
        else:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024,
            )
            try:
                stream.start_stream()
                read(stream)
            finally:
                stream.stop_stream()
                stream.close()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(e)
    finally:
        if not sc:
            p.terminate()


def start_microphone_input(
    sample_rate, sc=False, speaker=False, thread=False, queue_size=2
):
    que = mp.Queue(maxsize=queue_size)
    ready = mp.Event()
    pause = mp.Event()
    fin = mp.Event()

    if thread:
        p = threading.Thread(
            target=capture_microphone,
            args=(que, ready, pause, fin, sample_rate, sc, speaker),
            daemon=True,
        )
    else:
        p = mp.Process(
            target=capture_microphone,
            args=(que, ready, pause, fin, sample_rate, sc, speaker),
            daemon=True,
        )
    p.start()

    # キャプチャスレッド起動待ち
    while p.is_alive():
        if ready.is_set():
            break

    if not p.is_alive():
        raise Exception("Fail to start microphone capture.")

    params = dict(
        p=p,
        que=que,
        pause=pause,
        fin=fin,
    )

    return params
