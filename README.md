[<img src="ailia-models_B_241211.png">](ABOUT_AINYAN.md)

The collection of pre-trained, state-of-the-art AI models.

# About ailia SDK

[ailia SDK](https://ailia.ai/en/sdk/) is a cross-platform, high-speed inference SDK for AI. It supports Windows, Mac, Linux, iOS, Android, Jetson, and Raspberry Pi with GPU acceleration via Vulkan and Metal. Bindings are available for C++, Python, Unity (C#), Kotlin, Rust, and Flutter.

# Why ailia SDK

|  | ailia SDK | ONNX Runtime |
|:---|:---:|:---:|
| GPU inference via Vulkan and Metal | ✓ | − |
| ailia Speech / Voice / LLM / Tokenizer / Tracker | ✓ | − |
| 400+ verified model library with sample code | ✓ | − |
| Non-OS / RTOS inference support | ✓ | − |
| Unity bindings and model collection | ✓ | △ |
| Model‑specific optimization | ✓ | △ |

△ = Supported but limited due to general-purpose implementation.

# How to use

To try on your computer:

[ailia MODELS tutorial](TUTORIAL.md)  
[ailia MODELS tutorial 日本語版](TUTORIAL_jp.md)  

If you would like to try without setting up your computer:

[Try now on Google Colaboratory](https://www.ailia.ai/launch_to_colab)  

# Documentation

[ailia SDK documentation](https://docs.ailia.ai/en/)  
[ailia MODELS deepwiki](https://deepwiki.com/ailia-ai/ailia-models)  

# Latest update

[See update history](https://github.com/ailia-ai/ailia-models/wiki)

# Models

418 models are available.

| | Category | Models | Sub categories |
|:---|:---|:---:|:---|
| [<img src="action_recognition/va-cnn/image/f-0.png" width=128px>](/action_recognition/) | [Action recognition](/action_recognition/) | 6 |  |
| [<img src="anomaly_detection/mahalanobisad/bottle_test_good_000.png" width=128px>](/anomaly_detection/) | [Anomaly detection](/anomaly_detection/) | 5 |  |
|  | [Audio language model](/audio_language_model/) | 1 |  |
|  | [Audio processing](/audio_processing/) | 39 | Audio classification, Music enhancement, Music generation, Noise reduction, Phoneme alignment, Pitch detection, Speaker diarization, Speech to text, Text to speech, Voice activity detection, Voice conversion |
| [<img src="autonomous_driving/segformer/output.png" width=128px>](/autonomous_driving/) | [Autonomous driving](/autonomous_driving/) | 3 |  |
| [<img src="background_removal/deep-image-matting/output.png" width=128px>](/background_removal/) | [Background removal](/background_removal/) | 11 |  |
| [<img src="crowd_counting/crowdcount-cascaded-mtl/result.png" width=128px>](/crowd_counting/) | [Crowd counting](/crowd_counting/) | 2 |  |
| [<img src="deep_fashion/fashionai-key-points-detection/output_blouse.png" width=128px>](/deep_fashion/) | [Deep fashion](/deep_fashion/) | 6 |  |
| [<img src="depth_estimation/fcrn-depthprediction/input_depth.png" width=128px>](/depth_estimation/) | [Depth estimation](/depth_estimation/) | 13 |  |
| [<img src="diffusion/latent-diffusion-txt2img/output.png" width=128px>](/diffusion/) | [Diffusion](/diffusion/) | 15 | Text to image, Text to audio, Others |
| [<img src="face_detection/mtcnn/output.jpg" width=128px>](/face_detection/) | [Face detection](/face_detection/) | 9 |  |
| [<img src="face_identification/facenet_pytorch/data/angelina_jolie.jpg" width=128px>](/face_identification/) | [Face identification](/face_identification/) | 5 |  |
| [<img src="face_recognition/mivolo/output.png" width=128px>](/face_recognition/) | [Face recognition](/face_recognition/) | 21 | Age gender estimation, Emotion recognition, Gaze estimation, Head pose estimation, Keypoint detection, Others |
| [<img src="face_restoration/gfpgan/out_03.png" width=128px>](/face_restoration/) | [Face restoration](/face_restoration/) | 2 |  |
| [<img src="face_swapping/deepfacelive/sample_results/frame_000001_res.png" width=128px>](/face_swapping/) | [Face swapping](/face_swapping/) | 3 |  |
| [<img src="feature_extraction/dinov3/output.png" width=128px>](/feature_extraction/) | [Feature extraction](/feature_extraction/) | 1 |  |
| [<img src="frame_interpolation/cain/sample_results/output_0.png" width=128px>](/frame_interpolation/) | [Frame interpolation](/frame_interpolation/) | 4 |  |
| [<img src="generative_adversarial_networks/pytorch-gan/output_anime.png" width=128px>](/generative_adversarial_networks/) | [Generative adversarial networks](/generative_adversarial_networks/) | 8 |  |
| [<img src="hand_detection/hand_detection_pytorch/CARDS_OFFICE_output.jpg" width=128px>](/hand_detection/) | [Hand detection](/hand_detection/) | 3 |  |
| [<img src="hand_recognition/hand3d/output.png" width=128px>](/hand_recognition/) | [Hand recognition](/hand_recognition/) | 5 |  |
| [<img src="image_captioning/image_captioning_pytorch/demo.jpg" width=128px>](/image_captioning/) | [Image captioning](/image_captioning/) | 3 |  |
| [<img src="image_classification/alexnet/clock.jpg" width=128px>](/image_classification/) | [Image classification](/image_classification/) | 27 | CNN, Transformer, Specific task |
| [<img src="image_inpainting/pytorch-inpainting-with-partial-conv/result.png" width=128px>](/image_inpainting/) | [Image inpainting](/image_inpainting/) | 5 |  |
| [<img src="image_manipulation/colorization/imgs_out/ansel_adams3_output.jpg" width=128px>](/image_manipulation/) | [Image manipulation](/image_manipulation/) | 17 |  |
| [<img src="image_quality_assessment/aesthetic-predictor/demo.jpg" width=128px>](/image_quality_assessment/) | [Image quality assessment](/image_quality_assessment/) | 1 |  |
| [<img src="image_restoration/nafnet/noise_output.png" width=128px>](/image_restoration/) | [Image restoration](/image_restoration/) | 1 |  |
| [<img src="image_segmentation/pytorch-fcn/result.jpg" width=128px>](/image_segmentation/) | [Image segmentation](/image_segmentation/) | 27 |  |
| [<img src="landmark_classification/places365/input.jpg" width=128px>](/landmark_classification/) | [Landmark classification](/landmark_classification/) | 2 |  |
| [<img src="line_segment_detection/dexined/output.png" width=128px>](/line_segment_detection/) | [Line segment detection](/line_segment_detection/) | 2 |  |
| [<img src="low_light_image_enhancement/agllnet/output.png" width=128px>](/low_light_image_enhancement/) | [Low light image enhancement](/low_light_image_enhancement/) | 2 |  |
|  | [Natural language processing](/natural_language_processing/) | 33 | Bert, Embedding, Error corrector, Grapheme to phoneme, Named entity recognition, Reranker, Sentence generation, Sentiment analysis, Summarize, Translation, Zero shot classification |
|  | [Network intrusion detection](/network_intrusion_detection/) | 2 |  |
| [<img src="neural_rendering/nerf/output.png" width=128px>](/neural_rendering/) | [Neural rendering](/neural_rendering/) | 2 |  |
|  | [NSFW detector](/nsfw_detector/) | 1 |  |
| [<img src="object_detection/yolox/output.jpg" width=128px>](/object_detection/) | [Object detection](/object_detection/) | 45 | CNN, Transformer, Specific target |
| [<img src="object_detection_3d/3d_bbox/output.png" width=128px>](/object_detection_3d/) | [Object detection 3d](/object_detection_3d/) | 6 |  |
| [<img src="object_tracking/deepsort/demo.gif" width=128px>](/object_tracking/) | [Object tracking](/object_tracking/) | 10 |  |
| [<img src="optical_flow_estimation/raft/output.png" width=128px>](/optical_flow_estimation/) | [Optical flow estimation](/optical_flow_estimation/) | 2 |  |
| [<img src="point_segmentation/pointnet_pytorch/output.png" width=128px>](/point_segmentation/) | [Point segmentation](/point_segmentation/) | 1 |  |
| [<img src="pose_estimation/openpose/output.png" width=128px>](/pose_estimation/) | [Pose estimation](/pose_estimation/) | 11 |  |
| [<img src="pose_estimation_3d/pose-hg-3d/output.png" width=128px>](/pose_estimation_3d/) | [Pose estimation 3d](/pose_estimation_3d/) | 7 |  |
| [<img src="road_detection/road-segmentation-adas/output.png" width=128px>](/road_detection/) | [Road detection](/road_detection/) | 9 |  |
| [<img src="rotation_prediction/rotnet/output.png" width=128px>](/rotation_prediction/) | [Rotation prediction](/rotation_prediction/) | 1 |  |
| [<img src="style_transfer/adain/output.png" width=128px>](/style_transfer/) | [Style transfer](/style_transfer/) | 6 |  |
| [<img src="super_resolution/srresnet/output.png" width=128px>](/super_resolution/) | [Super resolution](/super_resolution/) | 8 |  |
| [<img src="text_detection/east/output.png" width=128px>](/text_detection/) | [Text detection](/text_detection/) | 3 |  |
| [<img src="text_recognition/donut/cord_sample_receipt1.png" width=128px>](/text_recognition/) | [Text recognition](/text_recognition/) | 8 |  |
|  | [Time-series forecasting](/time_series_forecasting/) | 4 |  |
| [<img src="vehicle_recognition/vehicle-attributes-recognition-barrier/demo.png" width=128px>](/vehicle_recognition/) | [Vehicle recognition](/vehicle_recognition/) | 2 |  |
| [<img src="vision_language_model/llava/view.jpg" width=128px>](/vision_language_model/) | [Vision language model](/vision_language_model/) | 7 |  |
|  | [Commercial model](/commercial_model/) | 1 |  |

# Other platforms

Prototype with ailia MODELS (Python), then deploy to production.

- [unity version](https://github.com/ailia-ai/ailia-models-unity)
- [kotlin version](https://github.com/ailia-ai/ailia-models-kotlin)
- [c++ version](https://github.com/ailia-ai/ailia-models-cpp)
- [flutter version](https://github.com/ailia-ai/ailia-models-flutter)
- [rust version](https://github.com/ailia-ai/ailia-models-rust)

# Contact

- [Contact us](https://www.ailia.ai/en-contact-product)
- [Mail](mailto:contact@ailia.ai)
