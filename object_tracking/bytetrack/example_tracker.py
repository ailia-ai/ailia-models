#### example of ailia tracker with yolox detection ####

import argparse
import os
import random
import sys
import urllib.request

import cv2

import ailia
import ailia_tracker

import os, sys, platform
if (platform.system(), platform.machine().lower()) in {('Windows', 'arm64'), ('Windows', 'aarch64')}:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'util', 'woa'))
    import woa_imshow  # noqa: F401  (patches cv2 highgui on import)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "yolox_s.opt.onnx"
MODEL_PATH = "yolox_s.opt.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/yolox/"

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 640

TARGET_CATEGORY = 0  # person
COCO_CATEGORY_COUNT = 80
THRESHOLD = 0.4
IOU = 0.45

# ======================
# Argument Parser
# ======================

parser = argparse.ArgumentParser(description="ailia tracker example")
parser.add_argument(
    "-v", "--video", default="0",
    help="The input video path. If a number is specified, the webcam with the corresponding id is used."
)
parser.add_argument(
    "-e", "--env_id", type=int, default=ailia.ENVIRONMENT_AUTO,
    help="The backend environment id."
)
args = parser.parse_args()

# ======================
# Model download
# ======================

for file_name in [WEIGHT_PATH, MODEL_PATH]:
    if not os.path.exists(file_name):
        print("Downloading " + file_name + "...")
        urllib.request.urlretrieve(REMOTE_PATH + file_name, file_name)

# ======================
# Main
# ======================

def main():
    # detector initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        COCO_CATEGORY_COUNT,
        format=ailia.NETWORK_IMAGE_FORMAT_BGR,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
        env_id=args.env_id)
    detector.set_input_shape(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)

    # tracker initialize
    tracker = ailia_tracker.AiliaTracker()

    # video initialize
    if args.video.isdigit():
        capture = cv2.VideoCapture(int(args.video))
    else:
        capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print("[ERROR] \"" + args.video + "\" not found")
        sys.exit(1)

    id2color = {}

    while True:
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
            break

        # object detection
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detections = detector.run(img, THRESHOLD, IOU)

        # tracking
        for obj in detections:
            tracker.add_target(obj["category"], obj["prob"], obj["box"]["x"], obj["box"]["y"], obj["box"]["w"], obj["box"]["h"])
        tracker.compute()

        # display result
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        for obj in tracker.get_objects():
            if obj.category != TARGET_CATEGORY:
                continue
            if obj.id not in id2color:
                id2color[obj.id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            x = int(obj.x * frame_width)
            y = int(obj.y * frame_height)
            w = int(obj.w * frame_width)
            h = int(obj.h * frame_height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), id2color[obj.id], 2)
            cv2.putText(frame, str(obj.id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
        cv2.imshow("ailia tracker", frame)

    capture.release()
    cv2.destroyAllWindows()

    print("Program finished successfully.")


if __name__ == "__main__":
    main()
