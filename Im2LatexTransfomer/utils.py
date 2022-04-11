from munch import Munch
from PIL import Image
import numpy as np
import re
import cv2
import os

import torch


def parse_args(args, **kwargs):
    args = Munch({'epoch': 0}, **args)
    kwargs = Munch({'no_cuda': False, 'debug': False}, **kwargs)
    args.wandb = not kwargs.debug and not args.debug
    args.device = 'cuda' if torch.cuda.is_available() and not kwargs.no_cuda else 'cpu'
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [
        args.get('min_width', 32), args.get('min_height', 32)]
    if 'decoder_args' not in args or args.decoder_args is None:
        args.decoder_args = {}
    if 'model_path' in args:
        args.out_path = os.path.join(args.model_path, args.name)
        os.makedirs(args.out_path, exist_ok=True)
    return args


def pad(img: Image, divable=32):
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    data = np.array(img.convert('LA'))
    data = (data-data.min())/(data.max()-data.min())*255
    if data[..., 0].mean() > 128:
        # To invert the text to white
        gray = 255*(data[..., 0] < 128).astype(np.uint8)
    else:
        gray = 255*(data[..., 0] > 128).astype(np.uint8)
        data[..., 0] = 255-data[..., 0]

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    if rect[..., -1].var() == 0:
        im = Image.fromarray((rect[..., 0]).astype(np.uint8)).convert('L')
    else:
        im = Image.fromarray((255-rect[..., -1]).astype(np.uint8)).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, im.getbbox())
    return padded


def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' %
                      (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' %
                      (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s

