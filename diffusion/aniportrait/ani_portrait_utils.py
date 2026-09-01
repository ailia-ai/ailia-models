import numpy as np


def get_model_file_names() -> dict[str, dict[str, str]]:
    return {
        "aniportrait": {
            "a2m_model": {
                "weight": "a2m_model.onnx",
                "model": "a2m_model.onnx.prototxt",
            },
            "a2p_model": {
                "weight": "a2p_model.onnx",
                "model": "a2p_model.onnx.prototxt",
            },
            "denoising_unet": {
                "weight": "denoising_unet.onnx",
                "model": "denoising_unet.onnx.prototxt",
            },
            "encoder": {
                "weight": "encoder.onnx",
                "model": "encoder.onnx.prototxt",
            },
            "image_encoder": {
                "weight": "image_encoder.onnx",
                "model": "image_encoder.onnx.prototxt",
            },
            "pose_guider": {
                "weight": "pose_guider.onnx",
                "model": "pose_guider.onnx.prototxt",
            },
            "reference_unet": {
                "weight": "reference_unet.onnx",
                "model": "reference_unet.onnx.prototxt",
            },
        },
        "facemesh_v2": {
            "face_landmarks_detector": {
                "weight": "face_landmarks_detector.onnx",
                "model": "face_landmarks_detector.onnx.prototxt",
            },
            "face_detector": {
                "weight": "face_detector.onnx",
                "model": "face_detector.onnx.prototxt",
            },
        },
        "audio": {
            "wav2vec2feature_extractor": {
                "weight": "wav2vec2feature_extractor.onnx",
                "model": "wav2vec2feature_extractor.onnx.prototxt"
            }
        },
    }


def load_canonical_model() -> np.ndarray:
    vertices = []
    with open("canonical_model.obj", 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])

    return np.array(vertices)