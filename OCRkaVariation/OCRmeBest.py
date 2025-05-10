import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageEnhance
from pytesseract import Output
import os
import pyttsx3
from gtts import gTTS
import speech_recognition as sr
from pygame import mixer
import gtts
import time
import speech_recognition as sr

engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

folder_path = 'D:/c++/Python/OCRkaVariation/'

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            img_source = cv2.imread(os.path.join(folder_path, filename))
            img_source = Image.fromarray(img_source)

            img_source = img_source.resize((img_source.width * 2, img_source.height * 2), Image.BICUBIC)
            if img_source.mode == 'RGBA':
                img_source = img_source.convert('RGB')  # Convert to RGB if has alpha channel

            img_source = ImageEnhance.Brightness(img_source).enhance(1.2)
            img_source = ImageEnhance.Contrast(img_source).enhance(1.5)
            img_source = ImageEnhance.Sharpness(img_source).enhance(1.2)
            img_source.save(r"D:\c++\Python\OCRkaVariation\test3_OCRme2ndBest.jpg")

            def get_grayscale(image):
                return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

            def thresholding(image):
                return cv2.threshold(np.array(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            def opening(image):
                kernel = np.ones((5, 5), np.uint8)
                return cv2.morphologyEx(np.array(image), cv2.MORPH_OPEN, kernel)

            def canny(image):
                return cv2.Canny(np.array(image), 100, 200)

            gray = get_grayscale(img_source)
            thresh = thresholding(gray)
            opening_result = opening(gray)
            canny_result = canny(gray)

            all_texts = []

            for img, img_name in [(img_source, 'Original'), (gray, 'Grayscale'), (thresh, 'Threshold'), (opening_result, 'Opening'), (canny_result, 'Canny')]:
                d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='eng')
                n_boxes = len(d['text'])
                text = ""

                # if len(img.shape) == 2:
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                for i in range(n_boxes):
                    if int(d['conf'][i]) > 60:
                        text += d['text'][i] + " "

                all_texts.append(text.strip())

            best_text = max(all_texts, key=lambda x: (len(x), -all_texts.index(x)))

            print("Best Recognized Text:")
            print(best_text)
            # engine.say(best_text)
            # engine.waitandrun

        except Exception as e:
            print(f"Error processing image {filename}: {e}")
#============================================================================
            # Function to recognize speech
            import speech_recognition as sr
            import time
            import pygame

            tts = gtts.gTTS(text=text, lang='en')
            tts.save(r"D:\c++\Python\OCRkaVariation\temp.mp3")
            os.system("mpg321 temp.mp3")

            # Initialize the recognizer and the microphone
            r = sr.Recognizer()
            mic = sr.Microphone()

            # Initialize pygame mixer for audio playback
            pygame.mixer.init()


            # Function to play audio
            def play_audio():
                pygame.mixer.music.load(r"D:\c++\Python\OCRkaVariation\temp.mp3")  # replace "audio_file.mp3" with your audio file
                pygame.mixer.music.play()


            # Function to pause audio
            def pause_audio():
                pygame.mixer.music.pause()


            # Function to unpause audio
            def unpause_audio():
                pygame.mixer.music.unpause()


            # Main loop
            while True:
                # Recognize audio from the microphone
                with mic as source:
                    print("Listening...")
                    audio = r.listen(source)

                # Try to convert the audio to text
                try:
                    text = r.recognize_google(audio)
                    print("You said: " + text)

                    # Check if the text matches the command to play audio
                    if text.lower() == "play":
                        play_audio()

                    # Check if the text matches the command to pause audio
                    elif text.lower() == "pause":
                        pause_audio()

                    # Check if the text matches the command to unpause audio
                    elif text.lower() == "unpause":
                        unpause_audio()

                # If the speech recognition fails, print an error message
                except:
                    print("Error during recognition")


            # Convert text to speech

            text_to_speech(best_text)

            # Recognize speech commands


            #
            # # Text-to-Speech Conversion
            # tts = gTTS(text=best_text, lang='en')
            # tts.save(r"D:\c++\Python\OCRkaVariation\text.mp3")
            #
            # # Initialize Pygame mixer
            # mixer.init()
            # mixer.music.load(r'D:\c++\Python\OCRkaVariation\text.mp3')
            #

            #     except sr.UnknownValueError:
            #         print("Google Speech Recognition could not understand audio")
            #     except sr.RequestError as e:
            #         print("Could not request results from Google Speech Recognition service; {0}".format(e))



