import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import math
import random
from PIL import Image, ImageTk

class YOLOv8ToVOCConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8_obb to VOC Bounding Box Converter")
        self.selected_directory = ""
        self.image_list = []
        self.bbox_list = []
        self.current_index = 0
        self.label_colors = {}

        # Create and place the directory selection button
        self.select_dir_button = tk.Button(root, text="Select Directory", command=self.open_directory, bg="#F833FF")
        self.select_dir_button.pack(pady=10)

        # Label to display the selected directory
        self.directory_label = tk.Label(root, text="No directory selected")
        self.directory_label.pack(pady=10)

        # Create and place previous and next buttons
        self.navigation_frame = tk.Frame(root)
        self.navigation_frame.pack(pady=10)

        self.previous_button = tk.Button(self.navigation_frame, text="Previous", command=self.previous_image, bg="#3357FF")
        self.previous_button.pack(side=tk.LEFT, padx=20)

        self.next_button = tk.Button(self.navigation_frame, text="Next", command=self.next_image, bg="#3357FF")
        self.next_button.pack(side=tk.RIGHT, padx=20)

        # Image panel to display the image
        self.image_panel = tk.Label(root)
        self.image_panel.pack(pady=10)

        # Bind key events
        self.root.bind('<a>', self.previous_image)
        self.root.bind('<d>', self.next_image)

    def open_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.directory_label.config(text="Selected Directory: " + dir_path)
            self.selected_directory = dir_path
            self.process_directory()

    def process_directory(self):
        self.image_list = []
        self.bbox_list = []
        self.label_colors = {}

        image_files = [x for x in os.listdir(self.selected_directory) if x.endswith('.jpg')]
        text_files = [x for x in os.listdir(self.selected_directory) if x.endswith('.txt')]

        for img_file in image_files:
            bbox_file = img_file.replace('.jpg', '.txt')
            if bbox_file in text_files:
                self.image_list.append(img_file)
                with open(os.path.join(self.selected_directory, bbox_file), 'r') as f:
                    bboxes = []
                    for line in f.readlines():
                        parts = line.strip().split()
                        label = parts[0]
                        coords = list(map(float, parts[1:]))
                        bboxes.append((label, coords))
                        if label not in self.label_colors:
                            self.label_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    self.bbox_list.append(bboxes)

        if not self.image_list or not self.bbox_list:
            messagebox.showerror("Error", "No valid images or bounding boxes found in the directory!")
            return

        self.current_index = 0
        self.display_image()

    def yolo_v8_obb_to_voc(self, bbox, image_size):
        label, x1, y1, x2, y2, x3, y3, x4, y4 = bbox
        new_x1 = math.ceil(image_size[1] * x1)
        new_y1 = math.ceil(image_size[0] * y1)
        new_x2 = math.ceil(image_size[1] * x2)
        new_y2 = math.ceil(image_size[0] * y2)
        new_x3 = math.ceil(image_size[1] * x3)
        new_y3 = math.ceil(image_size[0] * y3)
        new_x4 = math.ceil(image_size[1] * x4)
        new_y4 = math.ceil(image_size[0] * y4)

        return label, new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4

    def display_image(self, event=None):
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(self.image_list):
            self.current_index = len(self.image_list) - 1

        img_file = self.image_list[self.current_index]
        bboxes = self.bbox_list[self.current_index]
        img_path = os.path.join(self.selected_directory, img_file)
        img = cv2.imread(img_path)
        if img is None:
            return
        image_size = img.shape[:2]  # Get image size (height, width)

        for bbox in bboxes:
            label, coords = bbox
            converted_bbox = self.yolo_v8_obb_to_voc([label] + coords, image_size)
            label, new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4 = converted_bbox
            points = np.array([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            color = self.label_colors.get(label, (0, 255, 0))  # Default to green if no color found
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
            cv2.putText(img, label, (new_x1, new_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert image to PhotoImage for Tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the image panel
        self.image_panel.config(image=img_tk)
        self.image_panel.image = img_tk

    def next_image(self, event=None):
        self.current_index += 1
        self.display_image()

    def previous_image(self, event=None):
        self.current_index -= 1
        self.display_image()

# Set up the Tkinter window
root = tk.Tk()
app = YOLOv8ToVOCConverter(root)
root.mainloop()
