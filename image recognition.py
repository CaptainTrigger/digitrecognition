import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('my_digit_recognition.model')

if os.path.isfile('digits/digit.png'):
    img = cv2.imread('digits/digit.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0

    predictions = model.predict(img)[0]

    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title("Input Image")
    plt.show()

    for i, prob in enumerate(predictions):
        print(f"Probability for digit {i}: {prob:.4f}")

    predicted_digit = np.argmax(predictions)
    print("I predict this number is a:", predicted_digit)

    cv2.imshow("test", img[0, :, :, 0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image file 'digits/digit.png' not found.")

    u