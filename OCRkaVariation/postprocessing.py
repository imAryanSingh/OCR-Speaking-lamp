import enchant

# Initialize the spell checker
spell_checker = enchant.Dict("en_US")  # Choose the appropriate language dictionary

# Example OCR result
ocr_result = "D:\c++\Python\OCRkaVariation\input_image.png"

# Tokenize the OCR result (assuming whitespace separation)
words = ocr_result.split()

# Correct misspelled words
corrected_words = [spell_checker.suggest(word)[0] if not spell_checker.check(word) else word for word in words]

# Reconstruct the corrected result
corrected_result = ' '.join(corrected_words)

print(corrected_result)
