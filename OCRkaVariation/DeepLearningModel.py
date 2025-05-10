import pytesseract
import cv2

# Read the image
image = cv2.imread('D:\c++\Python\OCRkaVariation\input_image.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR using Tesseract LSTM model
result = pytesseract.image_to_string(gray_image, config='--oem 1 --psm 6')

print(result)
