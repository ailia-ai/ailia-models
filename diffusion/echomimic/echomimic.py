import sys
import time

# logger
from logging import getLogger  # noqa

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from detector_utils import load_image  # noqa


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_REF_UNET_PATH = "reference_unet.onnx"
WEIGHT_DNS_UNET_PATH = "denoising_unet.onnx"
WEIGHT_VAE_ENC_PATH = "vae_encode.onnx"
WEIGHT_VAE_DEC_PATH = "vae_decode.onnx"
MODEL_REF_UNET_PATH = "reference_unet.onnx.prototxt"
MODEL_DNS_UNET_PATH = "denoising_unet.onnx.prototxt"
MODEL_VAE_ENC_PATH = "vae_encode.onnx.prototxt"
MODEL_VAE_DEC_PATH = "vae_decode.onnx.prototxt"
PB_DNS_UNET_PATH = "denoising_unet_weights.pb"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/echomimic/"

IMAGE_PATH = "demo.png"
SAVE_IMAGE_PATH = "output.png"
IMAGE_SIZE = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("EchoMimic", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


class Audio2VideoPipeline:
    def __init__(
        self,
        vae_encode,
        vae_decode,
        reference_unet,
        denoising_unet,
    ):
        super().__init__()

        self.vae_encode = vae_encode
        self.vae_decode = vae_decode
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet

    def forward(
        self,
        image,
        audio_path,
        video_length,
        **kwargs,
    ):

        print("whisper_chunks:", whisper_chunks.shape)
        audio_frame_num = whisper_chunks.shape[0]
        audio_fea_final = np.expand_dims(whisper_chunks, axis=0)
        print("audio_fea_final:", audio_fea_final.shape)
        video_length = min(video_length, audio_frame_num)
        if video_length < audio_frame_num:
            audio_fea_final = audio_fea_final[:, :video_length, :, :]

        # c_face_locator_tensor = self.face_locator(face_mask_tensor)
        uc_face_locator_tensor = np.zeros_like(c_face_locator_tensor)
        face_locator_tensor = np.concatenate(
            [uc_face_locator_tensor, c_face_locator_tensor], axis=0
        )
        if not args.onnx:
            output = self.vae_encode.predict([ref_image])
        else:
            output = self.vae_encode.run(None, {"x": ref_image})
        ref_image_latents = output[0]
        ref_image_latents = ref_image_latents * 0.18215  # (b , 4, h, w)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps
        for t_i, t in enumerate(timesteps):
            noise_pred = np.zeros(
                (
                    latents.shape[0] * 2,
                    *latents.shape[1:],
                ),
                dtype=latents.dtype,
            )
            counter = np.zeros(
                (1, 1, latents.shape[2], 1, 1),
                dtype=latents.dtype,
            )

            # 1. Forward reference image
            if t_i == 0:
                timestep = np.zeros_like(t, dtype=np.float16)
                if not args.onnx:
                    output = self.reference_unet.predict([ref_image_latents, timestep])
                else:
                    output = self.reference_unet.run(
                        None,
                        {
                            "sample": ref_image_latents,
                            "timestep": timestep,
                        },
                    )
                _, *bank = output

            num_context_batches = len(context_queue)

            global_context = []
            for j in range(num_context_batches):
                global_context.append(context_queue[j : (j + 1)])

            for context in global_context:
                new_context = [
                    [0 for _ in range(len(context[c_j]))] for c_j in range(len(context))
                ]
                for c_j in range(len(context)):
                    for c_i in range(len(context[c_j])):
                        new_context[c_j][c_i] = (
                            context[c_j][c_i] + t_i * 2
                        ) % video_length

                latent_model_input = np.concatenate(
                    [latents[:, :, c] for c in new_context], axis=2
                ).repeat(2, axis=0)

                c_audio_latents = np.concatenate(
                    [audio_fea_final[:, c] for c in new_context]
                )
                audio_latents = np.concatenate(
                    [np.zeros_like(c_audio_latents), c_audio_latents], 0
                )

                # CFG-doubled banks
                bank_args = {
                    "bank_%d" % i: np.concatenate([bank[i], bank[i]], axis=0)
                    for i in range(len(bank))
                }
                if not args.onnx:
                    output = self.denoising_unet.predict(
                        [
                            latent_model_input,
                            np.array(t, dtype=np.float16),
                            audio_latents,
                            face_locator_tensor,
                            *bank_args.values(),
                        ]
                    )
                else:
                    bank_args = {"bank_%d" % i: bank[i] for i in range(len(bank))}
                    output = self.denoising_unet.run(
                        None,
                        {
                            "sample": latent_model_input,
                            "timestep": np.array(t, dtype=np.float16),
                            "audio_cond_fea": audio_latents,
                            "face_musk_fea": face_locator_tensor,
                            **bank_args,
                        },
                    )
                pred = output[0]

                for j, c in enumerate(new_context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                    counter[:, :, c] = counter[:, :, c] + 1

            # perform guidance
            guidance_scale = 2.5
            a = noise_pred / counter
            split_size = a.shape[0] // 2
            noise_pred_uncond, noise_pred_text = np.split(a, [split_size])
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )


def recognize(pipe):
    audio_path = "echomimic_en.wav"
    video_length = 1200

    logger.info("Start inference...")

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                video = pipe.forward(
                    img,
                    audio_path,
                    video_length,
                )
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
            video = pipe.forward(
                img,
                audio_path,
                video_length,
            )

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_REF_UNET_PATH, MODEL_REF_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DNS_UNET_PATH, MODEL_DNS_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_ENC_PATH, MODEL_VAE_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_DEC_PATH, MODEL_VAE_DEC_PATH, REMOTE_PATH)
    check_and_download_file(PB_DNS_UNET_PATH, REMOTE_PATH)

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        ref_unet = ailia.Net(
            MODEL_REF_UNET_PATH,
            WEIGHT_REF_UNET_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        denoise_unet = ailia.Net(
            MODEL_DNS_UNET_PATH,
            WEIGHT_DNS_UNET_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        vae_encode = ailia.Net(
            MODEL_VAE_ENC_PATH,
            WEIGHT_VAE_ENC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        vae_decode = ailia.Net(
            MODEL_VAE_DEC_PATH,
            WEIGHT_VAE_DEC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        ref_unet = onnxruntime.InferenceSession(
            WEIGHT_REF_UNET_PATH, providers=providers
        )
        dns_unet = onnxruntime.InferenceSession(
            WEIGHT_DNS_UNET_PATH, providers=providers
        )
        vae_encode = onnxruntime.InferenceSession(
            WEIGHT_VAE_ENC_PATH, providers=providers
        )
        vae_decode = onnxruntime.InferenceSession(
            WEIGHT_VAE_DEC_PATH, providers=providers
        )

    pipe = Audio2VideoPipeline(
        **dict(
            vae_encode=vae_encode,
            vae_decode=vae_decode,
            reference_unet=ref_unet,
            denoising_unet=denoise_unet,
        )
    )

    # generate
    recognize(pipe)


if __name__ == "__main__":
    main()
