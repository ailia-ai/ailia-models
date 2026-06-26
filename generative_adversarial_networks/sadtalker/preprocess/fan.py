import cv2
import numpy as np

def calculate_points(heatmaps):
    # change heatmaps to landmarks
    B, N, H, W = heatmaps.shape
    HW = H * W
    BN_range = np.arange(B * N)

    heatline = heatmaps.reshape(B, N, HW)
    indexes = np.argmax(heatline, axis=2)

    preds = np.stack((indexes % W, indexes // W), axis=2)
    preds = preds.astype(np.float64, copy=False)

    inr = indexes.ravel()

    heatline = heatline.reshape(B * N, HW)
    x_up = heatline[BN_range, inr + 1]
    x_down = heatline[BN_range, inr - 1]
    # y_up = heatline[BN_range, inr + W]

    if any((inr + W) >= 4096):
        y_up = heatline[BN_range, 4095]
    else:
        y_up = heatline[BN_range, inr + W]
    if any((inr - W) <= 0):
        y_down = heatline[BN_range, 0]
    else:
        y_down = heatline[BN_range, inr - W]

    think_diff = np.sign(np.stack((x_up - x_down, y_up - y_down), axis=1))
    think_diff *= .25

    preds += think_diff.reshape(B, N, 2)
    preds += .5
    return preds

class FAN():
    def __init__(self, face_align_net, use_onnx):
        self.face_align_net = face_align_net
        self.use_onnx = use_onnx

    def get_landmarks(self, img):
        H, W, _ = img.shape
        offset = W / 64, H / 64, 0, 0

        img = cv2.resize(img, (256, 256))
        inp = img[..., ::-1]
        inp = np.transpose(inp.astype(np.float32), (2, 0, 1))
        inp = np.expand_dims(inp / 255.0, axis=0)

        if self.use_onnx:
            outputs = self.face_align_net.run(None, {"input_image": inp})[0]
        else:
            outputs = self.face_align_net.run([inp])[0]
        out = outputs[:, :-1, :, :]

        pred = calculate_points(out).reshape(-1, 2)

        pred *= offset[:2]
        pred += offset[-2:]

        return pred
