import cv2
import numpy as np


class ImageEnhancer:

    @staticmethod
    def enhance(image):
        if image is None or image.size == 0:
            return image

        try:
            # Resize for performance
            image = cv2.resize(image, (640, 480))

            # CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)

            enhanced = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # Gamma correction
            gamma = 1.2
            inv = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)

            # FAST denoise (better than bilateral)
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, None, 3, 3, 7, 21
            )

            # Light sharpening
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            return enhanced

        except Exception as e:
            print(f"Enhancer error: {e}")
            return image