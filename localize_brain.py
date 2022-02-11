import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from rect import Rect

__brain_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def find_largest_polygon(mask, min_area=0.0005):
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = list(filter(lambda c: c.shape[0] > 2 and Polygon(c.squeeze()).area / (np.prod(mask.shape)) > min_area, ctrs))
    return ctrs


def detect_brain(image, min_threshold=5, min_area=0.0005):
    ret, thresh = cv2.threshold(image, min_threshold, 255, cv2.THRESH_BINARY)
    ctrs = find_largest_polygon(thresh, min_area)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, ctrs, color=255)
    mask = cv2.erode(mask, __brain_dilation_kernel, cv2.BORDER_CONSTANT)
    mask = cv2.dilate(mask, __brain_dilation_kernel, cv2.BORDER_CONSTANT)
    return mask, Rect(*cv2.boundingRect(cv2.findNonZero(mask))), ctrs


def main():
    image = cv2.imread('output/experiments/cache/129564675/65-thumbnail.jpg', cv2.IMREAD_GRAYSCALE)
    brain, bbox, poly = detect_brain(image)
    # cv2.rectangle(brain, *bbox.corners(), color=255)
    plt.imshow(brain, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
