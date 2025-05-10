"""
from PIL import Image
import pytesseract
import numpy as np
from pytesseract import Output
import cv2

# Set the tesseract path
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
filename = 'D:\c++\Python\OCRkaVariation\captured_image.png'
img = np.array(Image.open(filename))

# Perform OCR on the image
text = pytesseract.image_to_string(img)
print(text)

# Perform text localization and detection
results = pytesseract.image_to_data(img, output_type=Output.DICT)

# Extract the bounding box coordinates of the text region
for i in range(0, len(results["text"])):
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    text = results["text"][i]
    conf = int(results["conf"][i])
    if conf > 58:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

# Display the image with bounding boxes and text
cv2.imshow(" ", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageEnhance
from pytesseract import Output

try:
    img_source = cv2.imread('D:\c++\Python\OCRkaVariation\Viit.jpeg')
    img_source = Image.fromarray(img_source)

    img_source = img_source.resize((img_source.width * 2, img_source.height * 2), Image.BICUBIC)
    if img_source.mode == 'RGBA':
        img_source = img_source.convert('RGB')  # Convert to RGB if has alpha channel

    img_source = ImageEnhance.Brightness(img_source).enhance(1.2)
    img_source = ImageEnhance.Contrast(img_source).enhance(1.5)
    img_source = ImageEnhance.Sharpness(img_source).enhance(1.2)
    img_source.save(r"D:\c++\Python\OCRkaVariation\test3_OCRme2ndBest.jpg")

# img_source = np.array(img_source)
# img_source = cv2.resize(img_source, (img_source.shape[1] * 2, img_source.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
# #

    def get_grayscale(image):
    # return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)


    def thresholding(image):
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.threshold(np.array(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def opening(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(np.array(image), cv2.MORPH_OPEN, kernel)

    def canny(image):
    # return cv2.Canny(image, 100, 200)
        return cv2.Canny(np.array(image), 100, 200)

    gray = get_grayscale(img_source)
    thresh = thresholding(gray)
    opening_result = opening(gray)
    canny_result = canny(gray)

    all_texts = []

# Perform OCR on each processed image and accumulate the recognized text
    for img, img_name in [(img_source, 'Original'), (gray, 'Grayscale'), (thresh, 'Threshold'), (opening_result, 'Opening'), (canny_result, 'Canny')]:
        d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='eng')
        n_boxes = len(d['text'])
        text = ""

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                text += d['text'][i] + " "

        all_texts.append(text.strip())

# Compare results based on length and position
    best_text = max(all_texts, key=lambda x: (len(x), -all_texts.index(x)))

# Print the best recognized text
    print("Best Recognized Text:")
    print(best_text)

except Exception as e:
    print("Image not Clear")

"""
from picamera import PiCamera
import time
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
import cv2
import subprocess


    # Initialize camera
    camera = PiCamera()
    camera.resolution = (1920, 1080)  # Set the resolution

    # Start preview and let the camera warm up
    camera.start_preview()
    time.sleep(2)

    # Capture the image
    camera.capture("/home/aryan/Desktop/test3.jpg")
    subprocess.run(["espeak", "-ven+f3", '-v', 'en-in', '-p', '50', '-s', '190', '-g', '5',"Image captured successfully."])
    print("Image captured successfully")

    # Open the captured image
    image = Image.open("/home/aryan/Desktop/test3.jpg")
    camera.stop_preview()
    camera.close()

    # Resize and enhance the image
    improved_image = image.resize((image.width * 2, image.height * 2), Image.BICUBIC)
    if improved_image.mode == 'RGBA':
        improved_image = improved_image.convert('RGB')  # Convert to RGB if has alpha channel

    enhanced_image = ImageEnhance.Brightness(improved_image).enhance(1.2)
    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(1.5)
    enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(1.2)
    enhanced_image.save("/home/aryan/Desktop/test3_enhanced.jpg")

    # Perform OCR on the image
    
    img = cv2.GaussianBlur(cv2.imread("/home/aryan/Desktop/test3_enhanced.jpg", 0), (5, 5), 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(255-img_bin)
    print(text)
    # Filter out symbols and non-meaningful words
    text = ' '.join(word for word in text.split() if word.isalnum() and not word.isupper())
    
    print(text)
    if(text=='' or text==' '):
        subprocess.run(["espeak", "-ven+f3", '-v', 'en-in', '-p', '50', '-s', '190', '-g', '7',"Please set object properly, Image not found"])
    else:
        subprocess.run(["espeak", "-ven+f5", '-v', 'en-in', '-p', '50', '-s', '190', '-g', '7',"Filtered OCR Output: "+text])
    print("Filterd OCR Output")
    print(text)
    


    subprocess.run(["espeak", "-ven+f3", '-v', 'en-in', '-p', '50', '-s', '190', '-g', '7',"An error occurred:"+str(e)])
    print(e)
"""