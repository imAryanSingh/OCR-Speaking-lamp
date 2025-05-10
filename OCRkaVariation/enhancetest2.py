from PIL import Image, ImageEnhance

# Load the image
image = Image.open(r"D:\c++\Python\OCRkaVariation\captured_image.png")

# Check if the image has an alpha channel
if image.mode == 'RGBA':
    # If the image has an alpha channel, it needs to be converted to RGB
    image = image.convert('RGB')

# Enhance brightness
enhanced_image = ImageEnhance.Brightness(image).enhance(1.2)

# Enhance contrast
enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(1.2)

# Enhance sharpness
enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(1.2)

# Save the enhanced image
enhanced_image.save("enhanced_image.jpg")

print("Enhanced image saved as enhanced_image.jpg")
