import math
import sys
from logging import getLogger
from typing import Optional

import ailia
import cv2
import librosa
import numpy as np
import tqdm
from insightface.app import FaceAnalysis
from moviepy.editor import AudioFileClip, VideoClip
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

# import original modules
sys.path.append("../../util")

import df
from arg_utils import get_base_parser, get_savepath, update_parser
from detector_utils import load_image
from image_utils import normalize_image
from model_utils import check_and_download_file, check_and_download_models
from util_hallo import get_mask

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_VAE_ENC_PATH = "vae_encoder.onnx"
WEIGHT_VAE_DEC_PATH = "vae_decoder.onnx"
WEIGHT_AUDIO_ENC_PATH = "audio_encoder.onnx"
WEIGHT_REF_UNET_PATH = "reference_unet.onnx"
WEIGHT_DENOISE_PATH = "denoising_unet.onnx"
WEIGHT_FACE_LOC_PATH = "face_locator.onnx"
WEIGHT_AUDIO_PROJ_PATH = "audio_proj.onnx"
WEIGHT_IMAGE_PROJ_PATH = "image_proj.onnx"
MODEL_VAE_ENC_PATH = "vae_encoder.onnx.prototxt"
MODEL_VAE_DEC_PATH = "vae_decoder.onnx.prototxt"
MODEL_AUDIO_ENC_PATH = "audio_encoder.onnx.prototxt"
MODEL_REF_UNET_PATH = "reference_unet.onnx.prototxt"
MODEL_DENOISE_PATH = "denoising_unet.onnx.prototxt"
MODEL_FACE_LOC_PATH = "face_locator.onnx.prototxt"
MODEL_AUDIO_PROJ_PATH = "audio_proj.onnx.prototxt"
MODEL_IMAGE_PROJ_PATH = "image_proj.onnx.prototxt"
WEIGHT_DENOISE_PB_PATH = "denoising_unet_weights.pb"

WEIGHT_FACE_ANALYSIS_DET_PATH = "./face_analysis/models/scrfd_10g_bnkps.onnx"
WEIGHT_FACE_ANALYSIS_REG_PATH = "./face_analysis/models/glintr100.onnx"
WEIGHT_WAV2VEC_PATH = "./wav2vec/wav2vec2-base-960h/model.safetensors"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/hallo/"

IMAGE_SIZE = 512
SAMPLING_RATE = 16000
IMAGE_PATH = "demo.jpg"
WAV_PATH = "demo.wav"
SAVE_VIDEO_PATH = "output.mp4"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Hallo: Hierarchical Audio-Driven Visual Synthesis for Portrait Image Animation",
    IMAGE_PATH,
    SAVE_VIDEO_PATH,
)
parser.add_argument("--driving_audio", default=WAV_PATH, help="Input audio")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


def transform(img, width, height):
    img = np.array(
        Image.fromarray(img).resize((width, height), Image.Resampling.BILINEAR)
    )

    if img.ndim < 3:
        img = np.expand_dims(img, axis=2)

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = img.astype(np.float32)

    return img


def tensor_to_video(tensor, output_video_file, audio_source, fps=25):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_video_file (str): The file path where the output video will be saved.
        audio_source (str): The path to the audio file (WAV file) that contains the audio track to be added.
        fps (int): The frame rate of the output video. Default is 25 fps.
    """
    tensor = tensor.transpose(1, 2, 3, 0)  # convert to [f, h, w, c]
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # to [0, 255]

    def make_frame(t):
        # get index
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]

    new_video_clip = VideoClip(make_frame, duration=tensor.shape[0] / fps)
    audio_clip = AudioFileClip(audio_source).subclip(0, tensor.shape[0] / fps)
    new_video_clip = new_video_clip.set_audio(audio_clip)
    new_video_clip.write_videofile(output_video_file, fps=fps, audio_codec="aac")


# ======================
# Main functions
# ======================


class FaceAnimatePipeline:
    def __init__(
        self,
        vae_encoder,
        vae_decoder,
        audio_encoder,
        reference_unet,
        denoising_unet,
        face_locator,
        scheduler,
        image_proj,
        audio_proj,
        flg_onnx: bool = False,
    ):
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.audio_encoder = audio_encoder
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.scheduler = scheduler
        self.image_proj = image_proj
        self.audio_proj = audio_proj

        self.flg_onnx = flg_onnx

        self.vae_scale_factor = 8  # VAE downscaling factor

    def image_processor(self, img):
        height = width = IMAGE_SIZE
        img_rgb = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        _img = np.array(
            Image.fromarray(img_rgb).resize((width, height), Image.Resampling.BILINEAR)
        )
        _img = normalize_image(_img, normalize_type="127.5")
        _img = _img.transpose(2, 0, 1)  # HWC -> CHW
        _img = _img.astype(np.float32)
        pixel_values_ref_img = _img

        face_analysis_model_path = "./face_analysis/"
        face_analysis = FaceAnalysis(
            name="",
            root=face_analysis_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))

        # 2.1 detect face
        faces = face_analysis.get(img)
        if not faces:
            print(
                "No faces detected in the image. Using the entire image as the face region."
            )
            # Use the entire image as the face region
            face = {
                "bbox": [0, 0, img.shape[1], img.shape[0]],
                "embedding": np.zeros(512),
            }
        else:
            # Sort faces by size and select the largest one
            faces_sorted = sorted(
                faces,
                key=lambda x: (x["bbox"][2] - x["bbox"][0])
                * (x["bbox"][3] - x["bbox"][1]),
                reverse=True,
            )
            face = faces_sorted[0]  # Select the largest face

        """
        Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.
        """
        for _, model in face_analysis.models.items():
            if hasattr(model, "Dispose"):
                model.Dispose()

        # 2.2 face embedding
        face_emb = face["embedding"]

        # 2.3 render face mask
        face_region_ratio = 1.2
        (face_mask, sep_lip_mask, sep_background_mask, sep_face_mask) = get_mask(
            img_rgb, face_region_ratio
        )

        # 2.4 detect and expand lip, face mask
        face_mask = transform(face_mask, width, height)
        face_mask = np.repeat(face_mask, 3, axis=0)  # GRAY -> RGB
        pixel_values_face_mask = [
            transform(sep_face_mask, width // 8, height // 8),
            transform(sep_face_mask, width // 16, height // 16),
            transform(sep_face_mask, width // 32, height // 32),
            transform(sep_face_mask, width // 64, height // 64),
        ]
        pixel_values_lip_mask = [
            transform(sep_lip_mask, width // 8, height // 8),
            transform(sep_lip_mask, width // 16, height // 16),
            transform(sep_lip_mask, width // 32, height // 32),
            transform(sep_lip_mask, width // 64, height // 64),
        ]
        pixel_values_full_mask = [
            transform(sep_background_mask, width // 8, height // 8),
            transform(sep_background_mask, width // 16, height // 16),
            transform(sep_background_mask, width // 32, height // 32),
            transform(sep_background_mask, width // 64, height // 64),
        ]

        pixel_values_full_mask = [
            mask.reshape(1, -1) for mask in pixel_values_full_mask
        ]
        pixel_values_face_mask = [
            mask.reshape(1, -1) for mask in pixel_values_face_mask
        ]
        pixel_values_lip_mask = [mask.reshape(1, -1) for mask in pixel_values_lip_mask]

        return (
            pixel_values_ref_img,
            face_mask,
            face_emb,
            pixel_values_full_mask,
            pixel_values_face_mask,
            pixel_values_lip_mask,
        )

    def audio_processor(self, speech_array, clip_length: int = -1):
        wav2vec_model_path = "./wav2vec/wav2vec2-base-960h"
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_model_path, local_files_only=True
        )

        # extract wav2vec features
        audio_feature = np.squeeze(
            wav2vec_feature_extractor(
                speech_array, sampling_rate=SAMPLING_RATE
            ).input_values
        )
        fps = 25
        seq_len = math.ceil(len(audio_feature) / SAMPLING_RATE * fps)
        audio_length = seq_len

        if clip_length > 0 and seq_len % clip_length != 0:
            pad_len = (clip_length - seq_len % clip_length) % clip_length
            pad_len *= SAMPLING_RATE // fps
            audio_feature = np.pad(
                audio_feature, (0, pad_len), mode="constant", constant_values=0.0
            )
            seq_len += clip_length - seq_len % clip_length
        audio_feature = np.expand_dims(audio_feature, axis=0)

        seq_len = np.array(seq_len, dtype=np.int64)
        if not self.flg_onnx:
            output = self.audio_encoder.predict([audio_feature, seq_len])
        else:
            output = self.audio_encoder.run(
                None, {"audio_feature": audio_feature, "seq_len": seq_len}
            )
        hidden_state = output[0]

        audio_emb = hidden_state[1:]
        audio_emb = audio_emb.transpose(1, 0, 2)  # (b, s, d) -> (s, b, d)

        return audio_emb, audio_length

    def vae_image_processor(self, image, height, width):
        width, height = (x - x % self.vae_scale_factor for x in (width, height))

        N, C, *_ = image.shape
        resized = np.zeros((N, C, height, width), dtype=image.dtype)
        for n in range(N):
            for c in range(C):
                resized[n, c] = np.array(
                    Image.fromarray(image[n, c]).resize(
                        (width, height), Image.Resampling.BICUBIC
                    )
                )

        return resized

    def prepare_latents(
        self,
        batch_size: int,  # Number of videos to generate in parallel
        num_channels_latents: int,  # Number of channels in the latents
        width: int,  # Width of the video frame
        height: int,  # Height of the video frame
        video_length: int,  # Length of the video in frames
        generator: np.random.Generator,  # Random number generator for reproducibility
    ):
        """
        Prepares the initial latents for video generation.
        """
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = generator.normal(loc=0.0, scale=1.0, size=shape).astype(np.float16)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        """
        Decode the latents to produce a video.
        """
        video_length = latents.shape[2]  # f
        latents = latents / 0.18215

        # b c f h w -> (b f) c h w
        b, c, f, h, w = latents.shape
        latents = latents.transpose(0, 2, 1, 3, 4)
        latents = latents.reshape(b * f, c, h, w)

        video = []
        for i in tqdm.tqdm(range(latents.shape[0])):
            z = latents[i : i + 1]
            if not self.flg_onnx:
                output = self.vae_decoder.predict([z])
            else:
                output = self.vae_decoder.run(None, {"z": z})
            decoded = output[0]
            del output
            video.append(decoded)
        video = np.concatenate(video, axis=0)

        # (b f) c h w -> b c f h w
        b = video.shape[0] // video_length
        f = video_length
        c, h, w = video.shape[1:]
        video = video.reshape(b, f, c, h, w)
        video = video.transpose(0, 2, 1, 3, 4)

        video = np.clip(video / 2 + 0.5, 0, 1)
        return video

    def preprocess(self, image, speech_array, clip_length=16):
        # prepare inference data
        ## prepare source image, face mask, face embeddings
        (
            source_image_pixels,
            face_mask,
            face_emb,
            source_image_full_mask,
            source_image_face_mask,
            source_image_lip_mask,
        ) = self.image_processor(image)

        ## prepare audio embeddings
        audio_emb, audio_length = self.audio_processor(speech_array, clip_length)
        audio_emb = self.process_audio_emb(audio_emb)

        source_image_pixels = np.expand_dims(source_image_pixels, axis=0)
        face_mask = np.expand_dims(face_mask, axis=0)
        face_emb = face_emb.reshape(1, -1)
        source_image_full_mask = [
            np.tile(mask, (clip_length, 1)) for mask in source_image_full_mask
        ]
        source_image_face_mask = [
            np.tile(mask, (clip_length, 1)) for mask in source_image_face_mask
        ]
        source_image_lip_mask = [
            np.tile(mask, (clip_length, 1)) for mask in source_image_lip_mask
        ]

        face_mask = np.expand_dims(face_mask, axis=1).astype(
            dtype=np.float16
        )  # (bs, f, c, H, W)
        face_mask = np.repeat(face_mask, clip_length, axis=1)
        face_mask = face_mask.transpose(0, 2, 1, 3, 4)

        if not self.flg_onnx:
            output = self.face_locator.predict([face_mask])
        else:
            output = self.face_locator.run(None, {"conditioning": face_mask})
        face_mask = output[0]

        return (
            audio_emb,
            audio_length,
            source_image_pixels,
            face_mask,
            face_emb,
            source_image_full_mask,
            source_image_face_mask,
            source_image_lip_mask,
        )

    def process_audio_emb(self, audio_emb):
        """
        Process the audio embedding to concatenate with other tensors.
        """
        concatenated_tensors = []

        for i in range(audio_emb.shape[0]):
            vectors_to_concat = [
                audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)]
                for j in range(-2, 3)
            ]
            concatenated_tensors.append(np.stack(vectors_to_concat, axis=0))

        audio_emb = np.stack(concatenated_tensors, axis=0)

        return audio_emb

    def forward(
        self,
        ref_image,
        audio_tensor,
        face_emb,
        face_mask,
        pixel_values_full_mask,
        pixel_values_face_mask,
        pixel_values_lip_mask,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        motion_scale: np.ndarray,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[np.random.Generator] = None,
    ):
        if not self.flg_onnx:
            output = self.audio_proj.predict([audio_tensor])
        else:
            output = self.audio_proj.run(None, {"audio_embeds": audio_tensor})
        audio_tensor = output[0]

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        # prepare clip image embeddings
        clip_image_embeds = face_emb.astype(np.float16)
        if not self.flg_onnx:
            output = self.image_proj.predict([clip_image_embeds])
        else:
            output = self.image_proj.run(None, {"image_embeds": clip_image_embeds})
        encoder_hidden_states = output[0]

        if not self.flg_onnx:
            output = self.image_proj.predict([np.zeros_like(clip_image_embeds)])
        else:
            output = self.image_proj.run(
                None, {"image_embeds": np.zeros_like(clip_image_embeds)}
            )
        uncond_encoder_hidden_states = output[0]

        if do_classifier_free_guidance:
            encoder_hidden_states = np.concatenate(
                [uncond_encoder_hidden_states, encoder_hidden_states], axis=0
            )

        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            generator,
        )

        # Prepare ref image latents
        ## b f c h w -> (b f) c h w
        b, f, c, h, w = ref_image.shape
        ref_image_tensor = ref_image.reshape(b * f, c, h, w)
        ref_image_tensor = self.vae_image_processor(
            ref_image_tensor, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.astype(dtype=np.float32)

        if not self.flg_onnx:
            output = self.vae_encoder.predict([ref_image_tensor])
        else:
            output = self.vae_encoder.run(None, {"x": ref_image_tensor})
        ref_image_latents = output[0]
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
        ref_image_latents = ref_image_latents.astype(np.float16)

        face_mask = (
            np.concatenate([np.zeros_like(face_mask), face_mask], axis=0)
            if do_classifier_free_guidance
            else face_mask
        )
        pixel_values_full_mask = (
            [np.concatenate([mask] * 2) for mask in pixel_values_full_mask]
            if do_classifier_free_guidance
            else pixel_values_full_mask
        )
        pixel_values_face_mask = (
            [np.concatenate([mask] * 2) for mask in pixel_values_face_mask]
            if do_classifier_free_guidance
            else pixel_values_face_mask
        )
        pixel_values_lip_mask = (
            [np.concatenate([mask] * 2) for mask in pixel_values_lip_mask]
            if do_classifier_free_guidance
            else pixel_values_lip_mask
        )
        pixel_values_full_mask = [x.astype(np.float16) for x in pixel_values_full_mask]
        pixel_values_face_mask = [x.astype(np.float16) for x in pixel_values_face_mask]
        pixel_values_lip_mask = [x.astype(np.float16) for x in pixel_values_lip_mask]

        uncond_audio_tensor = np.zeros_like(audio_tensor)
        audio_tensor = np.concatenate([uncond_audio_tensor, audio_tensor], axis=0)
        audio_tensor = audio_tensor.astype(dtype=np.float16)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Forward reference image
                if i == 0:
                    # feedforward
                    if not self.flg_onnx:
                        output = self.reference_unet.predict(
                            [
                                np.tile(
                                    ref_image_latents,
                                    (2 if do_classifier_free_guidance else 1, 1, 1, 1),
                                ),
                                np.zeros_like(t),
                                encoder_hidden_states,
                            ]
                        )
                    else:
                        output = self.reference_unet.run(
                            None,
                            {
                                "sample": np.tile(
                                    ref_image_latents,
                                    (2 if do_classifier_free_guidance else 1, 1, 1, 1),
                                ),
                                "timestep": np.zeros_like(t),
                                "encoder_hidden_states": encoder_hidden_states,
                            },
                        )
                    _, *bank = output
                    del output

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    np.concatenate([latents] * 2)
                    if do_classifier_free_guidance
                    else latents
                )

                if not self.flg_onnx:
                    output = self.denoising_unet.predict(
                        [
                            latent_model_input,
                            np.array(t),
                            encoder_hidden_states,
                            audio_tensor,
                            face_mask,
                            *pixel_values_full_mask,
                            *pixel_values_face_mask,
                            *pixel_values_lip_mask,
                            motion_scale,
                            *bank,
                        ]
                    )
                else:
                    output = self.denoising_unet.run(
                        None,
                        {
                            "sample": latent_model_input,
                            "timestep": np.array(t),
                            "encoder_hidden_states": encoder_hidden_states,
                            "audio_embedding": audio_tensor,
                            "mask_cond_fea": face_mask,
                            **{
                                "full_mask_%d" % i: x
                                for i, x in enumerate(pixel_values_full_mask)
                            },
                            **{
                                "face_mask_%d" % i: x
                                for i, x in enumerate(pixel_values_face_mask)
                            },
                            **{
                                "lip_mask_%d" % i: x
                                for i, x in enumerate(pixel_values_lip_mask)
                            },
                            "motion_scale": motion_scale,
                            **{"bank_%d" % i: x for i, x in enumerate(bank)},
                        },
                    )
                noise_pred = output[0]
                del output

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, eta=eta)

                # call the callback, if provided
                if (
                    i == len(timesteps) - 1
                    or (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        return images

    def progress_bar(self, iterable=None, total=None):
        from tqdm.auto import tqdm

        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)


def recognize_from_video(pipe: FaceAnimatePipeline):
    image_path = args.input[0]
    driving_audio_path = args.driving_audio
    seed = args.seed

    # prepare input data
    image = load_image(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    speech_array, _ = librosa.load(driving_audio_path, sr=SAMPLING_RATE)

    logger.info("Start inference...")

    img_size = (512, 512)
    clip_length = 16
    motion_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    (
        audio_emb,
        audio_length,
        source_image_pixels,
        face_mask,
        face_emb,
        source_image_full_mask,
        source_image_face_mask,
        source_image_lip_mask,
    ) = pipe.preprocess(image, speech_array, clip_length=clip_length)

    n_motion_frames = 2
    generator = np.random.default_rng(seed)

    times = audio_emb.shape[0] // clip_length
    tensor_result = []
    for t in range(times):
        print(f"[{t+1}/{times}]")

        if len(tensor_result) == 0:
            # The first iteration
            motion_zeros = np.tile(source_image_pixels, (n_motion_frames, 1, 1, 1))
            motion_zeros = motion_zeros.astype(dtype=source_image_pixels.dtype)
            pixel_values_ref_img = np.concatenate(
                [source_image_pixels, motion_zeros], axis=0
            )  # concat the ref image and the first motion frames
        else:
            motion_frames = tensor_result[-1][0]
            motion_frames = motion_frames.transpose(1, 0, 2, 3)
            motion_frames = motion_frames[0 - n_motion_frames :]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.astype(dtype=source_image_pixels.dtype)
            pixel_values_ref_img = np.concatenate(
                [source_image_pixels, motion_frames], axis=0
            )  # concat the ref image and the motion frames
        pixel_values_ref_img = np.expand_dims(pixel_values_ref_img, axis=0)

        audio_tensor = audio_emb[
            t * clip_length : min((t + 1) * clip_length, audio_emb.shape[0])
        ]
        audio_tensor = np.expand_dims(audio_tensor, axis=0)
        audio_tensor = audio_tensor.astype(dtype=np.float16)

        videos = pipe.forward(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            face_emb=face_emb,
            face_mask=face_mask,
            pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=source_image_face_mask,
            pixel_values_lip_mask=source_image_lip_mask,
            width=img_size[0],
            height=img_size[1],
            video_length=clip_length,
            num_inference_steps=40,
            guidance_scale=3.5,
            generator=generator,
            motion_scale=motion_scale,
        )
        tensor_result.append(videos)
        del pixel_values_ref_img
        del audio_tensor
        del videos

    tensor_result = np.concatenate(tensor_result, axis=2)
    tensor_result = tensor_result.squeeze(0)
    tensor_result = tensor_result[:, :audio_length]

    output_file = get_savepath(args.savepath, "", ext=".mp4")
    logger.info(f"saved at : {output_file}")
    tensor_to_video(tensor_result, output_file, driving_audio_path)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_VAE_ENC_PATH, MODEL_VAE_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_DEC_PATH, MODEL_VAE_DEC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUDIO_ENC_PATH, MODEL_AUDIO_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_REF_UNET_PATH, MODEL_REF_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DENOISE_PATH, MODEL_DENOISE_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_AUDIO_PROJ_PATH, MODEL_AUDIO_PROJ_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_IMAGE_PROJ_PATH, MODEL_IMAGE_PROJ_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_DENOISE_PB_PATH, REMOTE_PATH)

    # for insightface
    check_and_download_file(WEIGHT_FACE_ANALYSIS_DET_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_FACE_ANALYSIS_REG_PATH, REMOTE_PATH)
    # wav2vec
    check_and_download_file(WEIGHT_WAV2VEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        vae_encoder = ailia.Net(
            MODEL_VAE_ENC_PATH,
            WEIGHT_VAE_ENC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        vae_decoder = ailia.Net(
            MODEL_VAE_DEC_PATH,
            WEIGHT_VAE_DEC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        audio_encoder = ailia.Net(
            MODEL_AUDIO_ENC_PATH,
            WEIGHT_AUDIO_ENC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        reference_unet = ailia.Net(
            MODEL_REF_UNET_PATH,
            WEIGHT_REF_UNET_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        denoising_unet = ailia.Net(
            MODEL_DENOISE_PATH,
            WEIGHT_DENOISE_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        face_locator = ailia.Net(
            MODEL_FACE_LOC_PATH,
            WEIGHT_FACE_LOC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        image_proj = ailia.Net(
            MODEL_IMAGE_PROJ_PATH,
            WEIGHT_IMAGE_PROJ_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        audio_proj = ailia.Net(
            MODEL_AUDIO_PROJ_PATH,
            WEIGHT_AUDIO_PROJ_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        vae_encoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_ENC_PATH, providers=providers
        )
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DEC_PATH, providers=providers
        )
        audio_encoder = onnxruntime.InferenceSession(
            WEIGHT_AUDIO_ENC_PATH, providers=providers
        )
        reference_unet = onnxruntime.InferenceSession(
            WEIGHT_REF_UNET_PATH, providers=providers
        )
        denoising_unet = onnxruntime.InferenceSession(
            WEIGHT_DENOISE_PATH, providers=providers
        )
        face_locator = onnxruntime.InferenceSession(
            WEIGHT_FACE_LOC_PATH, providers=providers
        )
        image_proj = onnxruntime.InferenceSession(
            WEIGHT_IMAGE_PROJ_PATH, providers=providers
        )
        audio_proj = onnxruntime.InferenceSession(
            WEIGHT_AUDIO_PROJ_PATH, providers=providers
        )

    scheduler = df.schedulers.DDIMScheduler.from_config(
        {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "steps_offset": 1,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing",
        },
    )

    pipe = FaceAnimatePipeline(
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        audio_encoder=audio_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        face_locator=face_locator,
        scheduler=scheduler,
        image_proj=image_proj,
        audio_proj=audio_proj,
        flg_onnx=args.onnx,
    )

    # generate
    recognize_from_video(pipe)


if __name__ == "__main__":
    main()
