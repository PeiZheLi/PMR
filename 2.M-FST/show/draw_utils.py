import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image

# COCO 17 points
point_name = ["left_scale", "right_scale", "front_pointer", "back_pointer"]

point_color = [(255, 0, 0), (255, 0, 0), (0, 0, 255),
               (0, 0, 255)]


def modify(x):
    if x < 0:
        return x + 360
    return x


def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 3,
                   draw_text: bool = True,
                   font: str = 'arial.ttf',
                   font_size: int = 10):
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)

        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i])
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    draw.line(keypoints.ravel()[-4:], fill=(0, 0, 255), width=3, joint=None)

    coordinate = keypoints.ravel()
    # print(coordinate)

    # x1 y1 0 1
    # x2 y2 2 3
    # x3 y3 4 5
    # x4 y4 6 7

    kl = np.arctan2(-(coordinate[1] - coordinate[7]), (coordinate[0] - coordinate[6])) * 180 / np.pi
    kr = np.arctan2(-(coordinate[3] - coordinate[7]), (coordinate[2] - coordinate[6])) * 180 / np.pi
    kp = np.arctan2(-(coordinate[5] - coordinate[7]), (coordinate[4] - coordinate[6])) * 180 / np.pi
    # print(kl * 180 / 3.1415926, kr * 180 / 3.1415926, kp * 180 / 3.1415926)

    kl = modify(kl)
    kr = modify(kr)
    kp = modify(kp)

    if kr - kl < 0:
        range = (kl - kr)
    else:
        range = (360 - kr + kl)

    met = (kl - kp)

    x = met / range * 100
    draw.text((0, 0), text=str(x)[:6], font=ImageFont.truetype('1.ttf', 48), fill=(255, 0, 0))

    return img
