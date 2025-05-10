import cv2
from pytesseract import pytesseract

# Read the image
image = cv2.imread('D:\c++\Python\OCRkaVariation\input_image.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply Gaussian blur for noise reduction
blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)

result = pytesseract.image_to_string(blurred_image, config='--oem 1 --psm 6')

print(result)

# Display the preprocessed image
cv2.imshow('Preprocessed Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
