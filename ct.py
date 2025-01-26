import uvicorn
from fastapi import FastAPI, File, UploadFile
import io
import os
import re

import json
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize
import cv2

app = FastAPI()

def process_image(image_path, output_size=(200, 200), threshold=128, epsilon_ratio=0.001):
    """
    Single function that:
      1. Converts an image to a simplified 2D binary matrix.
      2. Skeletonizes the binary matrix.
      3. Approximates contours from the skeletonized image.
      4. Computes bounding rectangle for all contour points.
      5. Prints boundaries and JSON data for contours.
      6. Returns that JSON data (as a Python dictionary).

    NOTE: The original plotting code has been removed as requested.
    """

    # --- Original helper function: image_to_simplified_binary_matrix ---
    def image_to_simplified_binary_matrix(image_path, output_size=(10, 10), threshold=128):
        """
        Convert an image to a simplified 2D binary matrix.
        """
        img = image_path.convert("L")  # Convert to grayscale
        img_resized = img.resize(output_size, Image.Resampling.LANCZOS)  # Resize image
        width, height = img_resized.size
        binary_matrix = [
            [1 if img_resized.getpixel((x, y)) < threshold else 0 for x in range(width)]
            for y in range(height)
        ]
        return binary_matrix

    # --- Original helper function: skeletonize_image ---
    def skeletonize_image(binary_matrix):
        """
        Apply skeletonization to a binary matrix (2D).
        """
        return skeletonize(binary_matrix)

    # --- Original helper function: approximate_contours ---
    def approximate_contours(binary_matrix, epsilon_ratio=0.01):
        """
        Approximate contours with points in a binary matrix using OpenCV.
        """
        binary_image = (binary_matrix * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty binary matrix with the same dimensions
        contour_matrix = np.zeros_like(binary_matrix, dtype=int)
        
        # Process each contour with points
        contour_points = []
        for contour in contours:
            epsilon = epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            contour_points.extend(approx)  # Add the points to contour_points
        
        return contour_matrix, contour_points

    # 1) Convert the image to a simplified binary matrix
    simplified_matrix = image_to_simplified_binary_matrix(
        image_path,
        output_size=output_size,
        threshold=threshold
    )

    # Convert to NumPy array for further processing
    binary_matrix_np = np.array(simplified_matrix)

    # 2) Skeletonize the binary matrix
    skeleton = skeletonize_image(binary_matrix_np)

    # 3) Approximate contours from the skeletonized image
    contour_matrix, contour_points = approximate_contours(skeleton, epsilon_ratio=epsilon_ratio)

    # 4) Compute bounding rectangle (left, right, top, bottom)
    if len(contour_points) > 0:
        all_points = np.vstack(contour_points)
        x, y, w, h = cv2.boundingRect(all_points)
        left   = x
        right  = x + w
        top    = y
        bottom = y + h
    else:
        left = right = top = bottom = 0

    # 5) Convert points to JSON-like structure
    contour_points_json = [
        {"x": int(point[0][0]), "y": 200 - int(point[0][1])}
        for point in contour_points
    ]

    data_for_json = {
        "boundaries": {
            "left":   int(left),
            "right":  int(right),
            "top":    int(top),
            "bottom": int(bottom)
        },
        "contours": contour_points_json
    }

    # Serialize to JSON string and print
    json_str = json.dumps(data_for_json, indent=2)

    # Return the Python dictionary (if you want to use it programmatically)
    return data_for_json

@app.post("/process-image")
async def process_image_endpoint(prompt: str = Form(...)):
    """
    Expects a JSON body with { "image_path": "/path/to/your/image.png" }.
    Returns processed contour & boundary data.
    """
    options = {
        "dog": "https://github.com/angelperedo97/hack/blob/main/dog.jpg",
        "boat": "https://github.com/angelperedo97/hack/blob/main/boat.png",
        "bird": "https://github.com/angelperedo97/hack/blob/main/bird.jpg",
        "flower":"https://github.com/angelperedo97/hack/blob/main/flower.png"
        "plant":"https://github.com/angelperedo97/hack/blob/main/plant.png"
    }

    option = re.sub(r'[^\w\s]', '', prompt)

    file_url = options[option]
    
    #img_bytes = await file.read()

    img = Image.open(io.BytesIO(file_url))
    result = process_image(img)
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ct:app", host="0.0.0.0", port=port)
