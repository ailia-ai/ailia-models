from typing import Dict, List, Optional, Union, Tuple

import cv2
import numpy as np

from constants import IMAGE_TOKEN, IMAGE_TOKEN_INDEX, CONVERSATION_START, ROLES, STOP_STR, BOS_TOKEN_ID, EOS_TOKEN_ID 


def pad_to_square(image: np.ndarray, background_color):
    height, width = image.shape[:2]
    if height == width:
        return image
    elif height > width:
        pad = (height - width) // 2
        image = np.pad(image, ((0, 0), (pad, height-width-pad), (0, 0)), mode='constant', constant_values=background_color)
    else:
        pad = (width - height) // 2
        image = np.pad(image, ((pad, width-height-pad), (0, 0), (0, 0)), mode='constant', constant_values=background_color)
    return image

class ImageProcessor():
    def __init__(self, 
        convert_to_rgb :bool = True,
        pad_to_square :bool = False,
        image_size :Union[int, Tuple[int, int]] = (336, 336),
        image_std :np.ndarray = np.array([0.26862954, 0.26130258, 0.27577711]),
        image_mean :np.ndarray = np.array([0.48145466, 0.4578275, 0.40821073]),
        rescale_factor : float = 1/255,
        resample :int = cv2.INTER_AREA,
        as_channel_first: bool = False,
        add_batch_dimension :bool = False
    ):
        self.convert_to_rgb = convert_to_rgb
        self.pad_to_square = pad_to_square
        self.size = image_size
        self.mean = image_mean
        self.std = image_std
        self.rescale_factor = rescale_factor
        self.resample = resample
        self.as_channel_first = as_channel_first
        self.add_batch_dimension = add_batch_dimension


    def resize(self, image):
        if self.pad_to_square:
            image = pad_to_square(image, 0)
        if isinstance(self.size, int):
            size = (self.size, self.size)
        else:
            size = self.size
        return cv2.resize(image, dsize = size, interpolation=self.resample)
    
    def __call__(self,image):
        image = image.astype(np.float32)
        if self.rescale_factor != 1.0:
            image *= self.rescale_factor
        if self.convert_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mean is not None and self.std is not None:
            image = (image - self.mean[None, None, :]) / self.std[None, None, :]
        if self.size is not None:
            image = self.resize(image)
        if self.as_channel_first:
            image = image.transpose(2, 0, 1)
        if self.add_batch_dimension:
            image = image[None, ...]
        return image
    

    
def simple_prompt_preprocessor_single_image(prompt):
    new_prompt = f'{CONVERSATION_START} {ROLES[0]}: {IMAGE_TOKEN}\n{prompt} {ROLES[1]}:'
    return new_prompt

def simple_prompt_preprocessor(prompt):
    new_prompt = f'{CONVERSATION_START} {ROLES[0]}: {prompt} {ROLES[1]}:'
    return new_prompt

def insert_separator(X, sep):
    # written using double list comprehension
    return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

def tokenize_w_image_token(prompt, tokenizer):
    # split the prompt by the image token
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(IMAGE_TOKEN)]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == BOS_TOKEN_ID:
        offset = 1
        input_ids.append(BOS_TOKEN_ID)

    # insert_separator interleaves the prompt_chunks with the image token index
    for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
        # append the input_ids (excluding the first token if it is the bos)
        input_ids.extend(x[offset:])

    return np.asarray(input_ids)

class llamaMultimodalInputProcesor():
    def __init__(self,
        vision_tower,
        mm_projector,
        token_embedder,
        dtype='float',
        onnx=False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.token_embedder = token_embedder
        assert dtype in ['float', 'half']
        self.dtype=np.float32 if dtype=='float' else np.float16

        self.onnx=onnx

    def encode_images(self, images):
        if self.onnx:
            image_features = self.vision_tower.run(None, {'images':images.astype(self.dtype)})[0]
            image_features = self.mm_projector.run(None, {'image_features':image_features.astype(self.dtype)})[0]
        else:
            image_features = self.vision_tower.predict(images)
            image_features = self.mm_projector.predict(image_features)
        return image_features

    def encode_images_ragged(self, images):
        concat_images = np.concatenate([image for image in images], axis=0)# (n, 3, 336, 336)
        image_features = self.encode_images(concat_images)# (n, 144, 512)
        split_sizes = [image.shape[0] for image in images]
        image_features = np.split(image_features, np.cumsum(split_sizes)[:-1], axis=0)# [(n1, 144, 512), (n2, 144, 512), ...]
        image_features = [image_faetures.flatten(0, 1) for image_faetures in image_features]# [(n1*144, 512), (n2*144, 512), ...]
        return image_features
    
    def _predict_embeds(self, input_ids):
        if self.onnx:
            return self.token_embedder.run(None, {'input_ids': input_ids})[0][0]
        else:
            return self.token_embedder.predict(input_ids)[0]
    
    def __call__(self, input_ids, images):
        """
        input_ids: prompts tokenized with proper image tokens inserted inplace of the IMAGE_TOKEN string
        images: 4 dimensional numpy array of images, or list of 4 dimensional numpy arrays.
                each image would be in place of the IMAGE_TOKEN string in the prompt
        len(images) should be equal to the number of IMAGE_TOKEN_INDEX in the input_ids
        """
        if isinstance(images, list) or images.ndim == 5:
            # outmost dimension (dim 0) is the total number of image features to be made
            # example:
            # images = [(2, 3, 336, 336), (4, 3, 336, 336), (3, 3, 336, 336)]
            # image_features will be [(288, 512), (576, 512), (432, 512)]
            image_features = self.encode_images_ragged(images)
        else:
            image_features = self.encode_images(images)
        # image_features: 3-dimensional, potentially ragged array of image features
        # first dimension is the number of image features, second dimension is the
        # length of the image features, and the third dimension is
        # the feature size (e.g. 512)
        # number of image features does not have to be the same es the batch size
        # since multiple images can be present in a single prompt

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):# iterate over batch
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                new_input_embeds.append(self._predict_embeds(cur_input_ids[None]))
                
                cur_image_idx += 1
                continue

            image_token_indices = np.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            # while there are image tokens in the input_ids
            while image_token_indices.size > 0:
            # ignoring config['tune_mm_mlp_adapter]
                assert cur_image_idx < len(image_features), "The number of images provided is less than the number of image tokens in the prompt"
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                
                # append the embeddings before the image token
                cur_new_input_embeds.append(self._predict_embeds(cur_input_ids[None, :image_token_start]))
                # append the image embeddings
                cur_new_input_embeds.append(cur_image_features)
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = np.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            # if there are any non-image tokens left
            if cur_input_ids.size > 0:
                cur_new_input_embeds.append(self._predict_embeds(cur_input_ids[None]))
    
            cur_new_input_embeds = np.concatenate(cur_new_input_embeds, axis=0)
            new_input_embeds.append(cur_new_input_embeds)
        assert cur_image_idx == len(image_features), "The number of images provided is more than the number of image tokens in the prompt"
        
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):# if the input_embeds are ragged (most likely)
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds = [np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant', constant_values=0) for x in new_input_embeds]
            new_attention_mask = [np.pad(np.ones(x.shape[0]), (0, max_len - x.shape[0]), mode='constant', constant_values=0) for x in new_input_embeds]
        else:
            new_attention_mask = [np.ones(x.shape[0]) for x in new_input_embeds]

        new_input_embeds = np.stack(new_input_embeds, axis=0)# shape of (n, l, d)
        new_attention_mask = np.stack(new_attention_mask, axis=0)# shape of (n, l)
        assert new_input_embeds.shape[:2] == new_attention_mask.shape
    
        return new_input_embeds, new_attention_mask.astype(np.int64)

