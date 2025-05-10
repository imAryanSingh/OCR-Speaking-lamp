import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageGrab, ImageEnhance
import numpy as np
import ctypes
import time

# Create a Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Display a warning message
messagebox.showwarning("Capture Image", "Prepare to capture image. Click 'OK' when ready.")

# Create a PIL image from the webcam
image = ImageGrab.grab()

# Save the image
image.save("captured_image.png")
messagebox.showinfo("Capture Image", "Image captured successfully!")

# Resize the image to improve quality
improved_image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)

# Enhance brightness and contrast
enhanced_image = ImageEnhance.Contrast(improved_image).enhance(1.5)
enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(1.2)

# Display the original and improved images
image.show(title="Original Image")
enhanced_image.show(title="Improved Image")

# Close the Tkinter window
root.destroy()
