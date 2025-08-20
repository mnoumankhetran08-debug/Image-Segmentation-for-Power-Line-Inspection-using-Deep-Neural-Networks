import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import onnxruntime as ort

# Load the ONNX model
session = ort.InferenceSession(r"D:\Power Line inspection\unet_model (3).onnx")

# GUI Setup
root = tk.Tk()
root.title("Power Line Defect Detector (ONNX)")
root.geometry("800x600")

# Function to upload and process image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        # Load and preprocess image (NHWC format)
        img = Image.open(file_path).convert('RGB').resize((256, 256))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

        # Run inference
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: img_array})[0]  # Shape: (1, 256, 256, 1)

        defect_prob = np.mean(pred)

        # Display processed image with overlay
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.imshow(pred[0, ..., 0], alpha=0.5, cmap='jet')  # Display segmentation mask
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().place(x=200, y=50)

        # Show defect detection result
        if defect_prob > 0.5:
            recommendation = f"⚠ Inspect for defects (Probability: {defect_prob:.2f})"
        else:
            recommendation = "✅ No significant defects detected"
        result_label.config(text=recommendation)

# Upload Button
upload_btn = tk.Button(root, text="Upload Power Line Image", command=upload_image)
upload_btn.place(x=50, y=50)

# Result Label
result_label = tk.Label(root, text="Upload an image to analyze.", wraplength=500, font=("Arial", 12))
result_label.place(x=200, y=400)

# Start GUI
root.mainloop()