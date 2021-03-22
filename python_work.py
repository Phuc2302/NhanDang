import cv2
import numpy as np
import cv_utils
import sheet
from functools import cmp_to_key

import matplotlib.pyplot as plt
import tensorflow as tf
import argparse


img = cv2.imread('D:/Study/AI/Python/NhanDang/Main/digits.jpg',cv2.IMREAD_GRAYSCALE)

# img_adaptive_binary = sheet.get_adaptive_binary_image(img)

# img_binary_yatzy_sheet = sheet.get_rotated_yatzy_sheet(img, img_adaptive_binary)

img_yatzy_sheet, img_binary_yatzy_sheet, img_binary_grid, yatzy_cells_bounding_rects = sheet.generate_yatzy_sheet(img)

# img_binary_gird, img_binary_sheet_only_digits = sheet.get_yatzy_grid(img_binary_yatzy_sheet)

# yatzy_cells_bounding_rects, yatzy_grid_bounding_rect = sheet.get_yatzy_cells_bounding_rects(img_binary_gird)

img_yatzy_cells = img_binary_grid.copy()
cv_utils.draw_bounding_rects(img_yatzy_cells, yatzy_cells_bounding_rects)


cv2.imshow('image' , img_binary_yatzy_sheet)
cv2.waitKey(0)
cv2.imshow('image' , img_binary_grid)
cv2.waitKey(0)
cv2.imshow('image' , img_yatzy_cells)
cv2.waitKey(0)

cv2.destroyAllWindows()
