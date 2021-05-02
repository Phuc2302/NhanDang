import tensorflow as tf
import argparse
import pytesseract
import numpy as np

import cv2
import cv_utils
import sheet
# import convert



print('Reading tensorflow model...')

predict_model = tf.keras.models.load_model('./models/model_tensorflow')
predict_model.summary()

# img_path = 'D:/Study/AI/Python/NhanDang/Main/digits.jpg'
img_path = './assets/test/2.jpg'
num_rows = 37

parser = argparse.ArgumentParser()

parser.add_argument("--num_rows", help="set num rows of the grid")
parser.add_argument("--img_path", help="specify path to input image")
parser.add_argument("--debug", help="specify debug to stop at show_window")

args = parser.parse_args()

if args.num_rows:
    num_rows = int(args.num_rows)

if args.debug:
    cv_utils.set_debug(bool(args.debug))

if args.img_path:
    img_path = args.img_path

# Step 1
print("Reading image from path", img_path)
input_img = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)

img_sheet, img_binary_sheet, img_binary_only_numbers, cells_bounding_rects = sheet.generate_sheet(input_img, num_rows_in_grid=num_rows)


img_cells = img_sheet.copy()
cv_utils.draw_bounding_rects(img_cells, cells_bounding_rects)
cv_utils.show_window('img_cells', img_cells)

digit_contours = cv_utils.get_external_contours(img_binary_only_numbers)


for i, cnt in enumerate(digit_contours):

    # Step 4
    digit_bounding_rect = cv2.boundingRect(cnt)
    x, y, w, h = digit_bounding_rect

    # Identify if and to which cell this bounding rect belongs to
    cell = sheet.validate_and_find_cell(cells_bounding_rects, digit_bounding_rect)
    if cell is None:
        continue

    # Black/white binary version of the roi
    roi = img_binary_sheet[y:y + h, x:x + w]

    roi_fit_20x20 = 20 / max(roi.shape[0], roi.shape[1])

    # Resize preserving binary format with INTER_NEAREST
    roi = cv2.resize(roi, None, fx=roi_fit_20x20, fy=roi_fit_20x20, interpolation=cv2.INTER_NEAREST)

    roi_background = np.zeros((28, 28), dtype=roi.dtype)
    # Place the digit in the roi_background, 4 from top and 4 from left.
    roi_background[4:4 + roi.shape[0], 4:4 + roi.shape[1]] = roi

    # Save the original roi
    cv2.imwrite("./assets/roi/original/roi_" +
                str(i) + ".png", roi_background)

    # Get the translation based on center of mass
    delta_x, delta_y = cv_utils.get_com_shift(roi_background)

    # Shift
    roi_background = cv_utils.shift_by(roi_background, delta_x, delta_y)
    cv2.imwrite("./assets/roi/shifted/roi_" +
                str(i) + ".png", roi_background)

    # Preprocess for prediction
    roi_background = roi_background - 127.5
    roi_background /= 127.5

    # Step 5
    # Log loss probabilities from our softmax classifier
    prediction = predict_model(np.reshape(roi_background, (1, 28, 28, 1)))

    predicted_digit = np.argmax(prediction)

    # print(predicted_digit)

    # Mark them on the image
    cv2.rectangle(img_sheet, (x, y), (x + w, y + h), (100, 10, 100), 1)

    cv2.putText(img_sheet, str(predicted_digit), (x + int(w / 2) + 40, y + int(h / 30) + 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                (0, 0, 255), 2, cv2.LINE_AA)


    import csv

    # import os
    #
    # if os.path.exists('./assets/output_data/data.csv'):
    #     os.remove('./assets/output_data/data.csv')
    # else:
    #     print("The file does not exists")

    arr = np.array([predicted_digit])

    outfile = open('./assets/output_data/data.csv', 'a')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: [x], reversed(arr)))
    outfile.close()


cv_utils.show_window('img_sheet', img_sheet, debug=True)
cv_utils.show_window('img_binary_sheet', img_binary_sheet)
cv_utils.show_window('img_binary_only_numbers', img_binary_only_numbers)
