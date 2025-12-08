from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Build a text prompt by prepending image tokens and a BOS token.

    Args:
        prefix_prompt (str): Original user/text prompt.
        bos_token (str): Beginning-of-sequence token from the tokenizer.
        image_seq_len (int): Number of image tokens to prepend.
        image_token (str): The special image token string.

    Returns:
        str: Combined prompt string with image tokens + BOS + original prompt.
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:

    """
    Rescale image pixel values by a constant factor.

    Args:
        image (np.ndarray): Input image array.
        scale (float): Multiplicative factor (e.g., 1/255.0 to map [0,255] → [0,1]).
        dtype (np.dtype): Output data type.

    Returns:
        np.ndarray: Rescaled image as `dtype`.
    """
    
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:


    """
    Resize a PIL image to the given (height, width).

    Args:
        image (PIL.Image.Image): Input image.
        size (Tuple[int, int]): Target (height, width).
        resample (Image.Resampling, optional): Resampling strategy (e.g., BICUBIC).
        reducing_gap (int, optional): Optimization hint for PIL.

    Returns:
        PIL.Image.Image: Resized image.
    """

    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:

    """
    Normalize an image tensor using channel-wise mean and std.

    Args:
        image (np.ndarray): Image array of shape (..., C) or (..., H, W, C).
        mean (float or Iterable[float]): Mean value(s) per channel.
        std (float or Iterable[float]): Std deviation value(s) per channel.

    Returns:
        np.ndarray: Normalized image.
    """


    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:


    """
    Convert a list of PIL images into normalized CHW numpy arrays.

    Pipeline per image:
        1. Resize to (height, width).
        2. Convert to numpy array (H, W, C).
        3. Rescale pixel values (e.g., /255).
        4. Normalize using mean/std.
        5. Transpose to (C, H, W).

    Args:
        images (List[PIL.Image.Image]): Input images.
        size (Tuple[int, int]): Target (height, width).
        resample (Image.Resampling, optional): Resampling mode for resize.
        rescale_factor (float, optional): Factor to multiply pixel values by.
        image_mean (float or List[float], optional): Channel-wise mean for normalization.
        image_std (float or List[float], optional): Channel-wise std for normalization.

    Returns:
        List[np.ndarray]: List of processed images with shape (C, H, W).
    """

    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:

    """
    Prepares text and images for a PaliGemma-style vision-language model.

    Responsibilities:
        - Extend tokenizer with image/loc/seg tokens.
        - Preprocess images to normalized CHW tensors.
        - Build input prompts with image tokens and BOS.
        - Tokenize prompts and return tensors ready for the model.

    Attributes:
        IMAGE_TOKEN (str): Special token used to represent image patches.
        image_seq_length (int): Number of image tokens per image.
        image_size (int): Target size (square) for image preprocessing.
        image_token_id (int): ID of IMAGE_TOKEN in tokenizer vocabulary.
        tokenizer: Tokenizer instance (e.g., a HuggingFace tokenizer).
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        """
        Initialize the processor and extend the tokenizer.

        Args:
            tokenizer: Tokenizer object to be extended with image and extra tokens.
            num_image_tokens (int): Number of image tokens to prepend per image.
            image_size (int): Target image size (images will be resized to image_size x image_size).
        """

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:


    """
        Process a single text prompt and a single image into model-ready tensors.

        Args:
            text (List[str]): List containing exactly one text prompt.
            images (List[PIL.Image.Image]): List containing exactly one image.
            padding (str): Padding strategy passed to tokenizer (e.g., "longest").
            truncation (bool): Whether to truncate text sequences.

        Returns:
            dict: Dictionary with:
                - "pixel_values": torch.Tensor of shape (1, C, H, W)
                - "input_ids": torch.Tensor of token ids
                - "attention_mask": torch.Tensor attention mask (and any other tokenizer outputs)
        """

        
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data