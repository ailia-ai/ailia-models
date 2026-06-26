import itertools
import sys
import time
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from detector_utils import load_image  # noqa

# Import MobileVLM modules
from preprocessing_numpy import ImageProcessor, llamaMultimodalInputProcesor, tokenize_w_image_token, simple_prompt_preprocessor_single_image
from constants import IMAGE_TOKEN, CONVERSATION_START, STOP_STR, BOS_TOKEN_ID, EOS_TOKEN_ID

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/mobilevlm/"

IMAGE_PATH = "demo.jpg"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("MobileVLM", IMAGE_PATH, None, large_model = True)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="Who is the author of this book? Answer in a single sentence.",
    help="prompt",
)

parser.add_argument(
    "--model_size",
    type=str,
    default='1.7B',
    help='parameter count of MobileVLM'
)

parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
#parser.add_argument(
#    "--fp16", action="store_true", help="use fp16 model (default : fp32 model)."
#)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="temperature for token selection",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.95,
    help="top_p from generation_config.json",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=1,
    help="top_k from generation_config.json",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=256,
    help="max_length for generation",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Model selection
# ======================

postfix = {
    '1.7B': '17',
    '3B': '3'
}

# Define model components
MODEL_COMPONENTS = [
    'token_embedder',
    'vision_tower',
    'mm_projector',
    'mobilellama_kvc',  # Always use KVC model
]

# Define paths for each component
MODEL_PATHS = {}
WEIGHT_PATHS = {}

for component in MODEL_COMPONENTS:
    # Vision tower are shared between model sizes
    if component in ['vision_tower']:
        weight_filename = f"{component}.onnx"
        model_filename = f"{component}.onnx.prototxt"
    else:
        # Other components are specific to model size
        weight_filename = f"{component}_{postfix[args.model_size]}.onnx"
        model_filename = f"{component}_{postfix[args.model_size]}.onnx.prototxt"
    
    WEIGHT_PATHS[component] = weight_filename
    MODEL_PATHS[component] = model_filename

for key, value in MODEL_PATHS.items():
    MODEL_PATHS[key] = value
for key, value in WEIGHT_PATHS.items():
    WEIGHT_PATHS[key] = value

# ======================
# KV Cache Configuration
# ======================

# KV cache dimensions based on model size
KVC_CONFIG = {
    '1.7B': {
        'num_layers': 24,
        'num_heads': 32,
        'head_dim': 128,
    },
    '3B': {
        'num_layers': 32,
        'num_heads': 32,
        'head_dim': 80,
    }
}

# ======================
# Model Loading
# ======================

def load_models(env_id=0):
    logger.info("Loading models...")
    models = {}

    # Check and download model files
    for component in MODEL_COMPONENTS:
        model_path = MODEL_PATHS[component]
        weight_path = WEIGHT_PATHS[component]
        check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # Model weights
    check_and_download_file(
        WEIGHT_PATHS["mobilellama_kvc"][:-5] + "_weights.pb",
        REMOTE_PATH
    )
    
    # Additional files for tokenizer
    tokenizer_path = f"tokenizer"
    
    memory_mode = ailia.get_memory_mode(
        reduce_constant=True,
        ignore_input_with_initializer=True,
        reduce_interstage=False,
        reuse_interstage=True,
    )
    
    for component in MODEL_COMPONENTS:
        model_path = MODEL_PATHS[component]
        weight_path = WEIGHT_PATHS[component]
        logger.info(f"Loading {component}...")
        
        if not args.onnx:
            models[component] = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=memory_mode)
        else:
            import onnxruntime
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            models[component] = onnxruntime.InferenceSession(weight_path, providers=providers)
    
    # Load tokenizer
    if args.disable_ailia_tokenizer:
        try:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(f"./{tokenizer_path}")
        except ImportError:
            logger.error("transformers is not installed. Please install it or use ailia tokenizer.")
            sys.exit(1)
    else:
        try:
            from ailia_tokenizer import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(f"./{tokenizer_path}")
        except ImportError:
            logger.error("ailia_tokenizer is not installed. Please install it or use --disable_ailia_tokenizer.")
            sys.exit(1)
    
    models["tokenizer"] = tokenizer
    return models


# ======================
# Image Processing
# ======================

def preprocess_images(image_paths):
    """Preprocess images for the MobileVLM model."""
    logger.info(f"Processing {len(image_paths)} images...")
    
    images = []
    for image_path in image_paths:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            sys.exit(1)
        
        # Process image
        processor = ImageProcessor(
            convert_to_rgb=True,
            pad_to_square=True,
            image_size=(336, 336),
            rescale_factor=1/255,
            resample=cv2.INTER_AREA,
            as_channel_first=True,
            add_batch_dimension=False
        )
        processed_img = processor(img)
        images.append(processed_img)
    
    # Stack all images into a single batch
    if len(images) > 0:
        images = np.stack(images, axis=0)
    else:
        # Create dummy image if no images are provided
        images = np.zeros((1, 3, 336, 336), dtype=np.float32)
    
    return images


# ======================
# Text Generation
# ======================

def sample_next_token(logits, temperature=0.7, top_p=0.9, top_k=10):
    """Apply temperature, top-k and top-p filtering to logits and sample next token."""
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    
    # Apply softmax to get probabilities
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    
    # Apply top-k filtering
    if top_k > 0:
        top_k = min(top_k, probs.shape[0])
        indices_to_remove = np.argsort(probs)[:-top_k]
        probs[indices_to_remove] = 0
        
    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to ensure we keep the first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
    
    # Renormalize probabilities
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        # If all probabilities were filtered out, reset to uniform
        probs = np.ones_like(probs) / len(probs)
    
    # Sample from filtered distribution
    next_token = np.random.choice(len(probs), p=probs)
    return next_token

def initialize_kv_cache(batch_size=1, model_size='1.7B'):
    """Initialize an empty KV cache based on model size."""
    config = KVC_CONFIG[model_size]
    
    kv_cache = np.zeros((
        config['num_layers'],
        2,
        batch_size,
        config['num_heads'],
        0,
        config['head_dim']
    #), dtype=np.float16 if args.fp16 else np.float32)
    ), dtype=np.float32)

    return kv_cache

def generate_text(models, prompt, images, max_len=256, temperature=0.7, top_p=0.9, top_k=10):
    """Generate text based on the prompt and images."""
    tokenizer = models["tokenizer"]
    
    # Format prompt for conversation
    formatted_prompt = simple_prompt_preprocessor_single_image(prompt)
    logger.info(f"Formatted prompt: {formatted_prompt}")
    
    # Tokenize prompt with image token
    input_ids = tokenize_w_image_token(formatted_prompt, tokenizer)
    input_ids = np.expand_dims(input_ids, axis=0)  # Add batch dimension
    
    # Create multimodal processor to handle image and text inputs
    multimodal_processor = llamaMultimodalInputProcesor(
        models['vision_tower'],
        models['mm_projector'],
        models['token_embedder'],
        #dtype='half' if args.fp16 else 'float',
        dtype='float',
        onnx=args.onnx
    )
    
    # Process inputs (combine text and image embeddings)
    input_embeds, attention_mask = multimodal_processor(input_ids, images)
    
    # Setup for text generation
    generated_tokens = []
    batch_size = 1
    
    # Initialize KV cache
    kv_cache = initialize_kv_cache(batch_size, args.model_size)
    model_key = 'mobilellama_kvc'
    
    # Generate text token by token
    for i in range(max_len):
        # For first token, use full input_embeds; for subsequent tokens, just the new token
        if i > 0:
            # Get embedding for the last generated token
            if args.onnx:
                next_token_embed = models['token_embedder'].run(
                    None, 
                    {'input_ids': np.array([[generated_tokens[-1]]], dtype=np.int64)}
                )[0]
            else:
                next_token_embed = models['token_embedder'].predict(
                    np.array([generated_tokens[-1]], dtype=np.int64)
                )
            
            # Just use the new token's embedding
            current_input_embeds = next_token_embed.reshape(1, 1, -1)
            
            # Update attention mask
            attention_mask = np.concatenate([
                attention_mask,
                np.ones((1, 1), dtype=np.int64)
            ], axis=1)
        else:
            current_input_embeds = input_embeds
        
        # Forward pass through the model
        if args.onnx:
            inputs = {
                'attention_mask': attention_mask,
                'past_key_values': kv_cache,
                'input_embeds': current_input_embeds
            }
            outputs = models[model_key].run(None, inputs)
            logits, kv_cache = outputs[0], outputs[1]
        else:
            models[model_key].set_input_blob_data(current_input_embeds, models[model_key].find_blob_index_by_name("input_embeds"))
            models[model_key].set_input_blob_data(attention_mask, models[model_key].find_blob_index_by_name("attention_mask"))
            models[model_key].set_input_blob_data(kv_cache, models[model_key].find_blob_index_by_name("past_key_values"))
            models[model_key].update()
            logits = models[model_key].get_blob_data(models[model_key].find_blob_index_by_name("logits"))
            kv_cache = models[model_key].get_blob_data(models[model_key].find_blob_index_by_name("past_key_values_out"))
        
        # Get logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Sample next token
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        # Append token to generated sequence
        generated_tokens.append(next_token)  # <-- FIX: append before decoding

        # Incremental print with space handling
        if i == 0:
            chunk = tokenizer.decode([next_token], skip_special_tokens=True)
            print(chunk, end='', flush=True)
        else:
            full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            prev_text = tokenizer.decode(generated_tokens[:-1], skip_special_tokens=True)
            new_text = full_text[len(prev_text):]
            print(new_text, end='', flush=True)

        # Check for end of sequence token
        if next_token == EOS_TOKEN_ID:
            break
    
    # Decode generated tokens
    print('\n')
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text


# ======================
# Main Function
# ======================

def main():
    """Main function to process images and generate text."""
    # Load models
    models = load_models(env_id=args.env_id)
    
    # Process images
    if not args.input:
        logger.error("No input images provided. Please specify at least one image with --input.")
        sys.exit(1)
    
    # Preprocess images
    images = preprocess_images(args.input)
    
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time = 0
        
        for i in range(args.benchmark_count):
            start = time.time()
            
            output = generate_text(
                models,
                args.prompt,
                images,
                max_len=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            
            end = time.time()
            duration = (end - start) * 1000  # Convert to ms
            logger.info(f"\tProcessing time {i+1}/{args.benchmark_count}: {duration:.2f} ms")
            
            if i > 0:  # Skip the first run for warm-up
                total_time += duration
        
        logger.info(f"\tAverage processing time: {total_time/(args.benchmark_count-1):.2f} ms")
    else:
        # Generate text
        logger.info(f"Prompt: {args.prompt}")
        output = generate_text(
            models,
            args.prompt,
            images,
            max_len=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
    
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
