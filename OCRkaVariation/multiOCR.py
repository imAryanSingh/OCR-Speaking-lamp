# import pytesseract
# from PIL import Image
#
# # Read the image
# image = Image.open('input_image.png')
#
# # Perform OCR using multiple engines
# results = []
# engines = ['tesseract', 'kraken', 'easyocr']
# for engine in engines:
#     result = pytesseract.image_to_string(image, config=f'--oem 3 --psm 6 --engine {engine}')
#     results.append(result)
#
# # Combine and select the most accurate result
# final_result = max(results, key=len)
#
# print(final_result)

import pytesseract
import cv2

# Read the image
image = cv2.imread('D:\c++\Python\OCRkaVariation\input_image.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR using multiple engines
results = []
engines = ['tesseract', 'kraken', 'easyocr']
for engine in engines:
    result = pytesseract.image_to_string(gray_image, config=f'--oem 3 --psm 6 --engine {engine}')
    results.append(result)

# Combine and select the most accurate result
final_result = max(results, key=len)

print(final_result)
