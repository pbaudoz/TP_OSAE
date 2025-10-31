import tkinter as tk
from tkinter import Label, Button, Entry
from PIL import Image, ImageTk
import cv2
import numpy as np
from visualisation import get_live_image, set_camera_roi, set_camera_exposure, capture_and_save_image
from Reference import process_and_save_images, draw_grid_around_spots, detect_spots

# --- Variables globales ---
show_grid = False
spots_centers = []
cell_size = 50

def assign_coordinates_from_file(filename='reference.txt'):
    coords = {}
    try:
        with open(filename,'r') as file:
            for line in file:
                if ":" in line:
                    key, val = line.split(":")
                    coords[key.strip()] = val.strip()
        return coords
    except Exception as e:
        print(f"❌ Erreur assignation coords : {e}")
        return None

def toggle_grid():
    global show_grid
    show_grid = not show_grid
    update_reference_image()

def update_image():
    image = get_live_image()
    if image is not None:
        img_to_show = image.copy()
        if show_grid and len(spots_centers) > 0:
            img_to_show = draw_grid_around_spots(img_to_show, spots_centers, cell_size)
        img_pil = Image.fromarray(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
        live_video_label.config(image=img_tk)
        live_video_label.image = img_tk
    root.after(10, update_image)

def update_reference_image():
    try:
        reference_image = cv2.imread('image_reference_centre.jpg')
        if reference_image is not None:
            if show_grid and len(spots_centers) > 0:
                reference_image = draw_grid_around_spots(reference_image, spots_centers, cell_size)
            img_pil = Image.fromarray(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img_pil)
            reference_image_label.config(image=img_tk)
            reference_image_label.image = img_tk
    except Exception as e:
        print(f"Erreur mise à jour image référence : {e}")
    root.after(1000, update_reference_image)

def acquire_new_reference_image():
    global spots_centers
    capture_and_save_image()
    process_and_save_images('image_reference.jpg', cell_size)
    # mettre à jour spots_centers depuis image zoomée
    gray_image = cv2.cvtColor(cv2.imread('image_reference_centre.jpg'), cv2.COLOR_BGR2GRAY)
    spots_centers = detect_spots(gray_image)
    update_reference_image()

def update_exposure():
    try:
        exposure_s = float(exposure_entry.get())
        exposure_us = exposure_s * 1e6
        set_camera_exposure(exposure_us)
        print(f"✅ Exposition réglée sur {exposure_s:.6f} s")
    except Exception as e:
        print(f"❌ Erreur réglage exposition : {e}")

# --- Interface ---
root = tk.Tk()
root.title("Caméra Live & Référence")

# Live
live_video_frame = tk.Frame(root)
live_video_frame.pack(side=tk.LEFT, padx=10, pady=10)
live_video_label = Label(live_video_frame)
live_video_label.pack()

# Référence
reference_frame = tk.Frame(root)
reference_frame.pack(side=tk.RIGHT, padx=10, pady=10)
reference_image_label = Label(reference_frame)
reference_image_label.pack()

# Boutons
acquire_button = Button(root, text="Acquérir nouvelle image référence", command=acquire_new_reference_image)
acquire_button.pack(pady=5)

grid_button = Button(root, text="Afficher / Masquer grille", command=toggle_grid)
grid_button.pack(pady=5)

exposure_label = Label(root, text="Exposition (s) :")
exposure_label.pack()
exposure_entry = Entry(root)
exposure_entry.insert(0,"0.02")
exposure_entry.pack()
exposure_button = Button(root, text="Appliquer", command=update_exposure)
exposure_button.pack(pady=5)

# --- Coordonnées zone utile ---
coords = assign_coordinates_from_file('reference.txt')
if coords:
    try:
        x_min, x_max = int(coords['x_min']), int(coords['x_max'])
        y_min, y_max = int(coords['y_min']), int(coords['y_max'])
        width, height = x_max - x_min, y_max - y_min
        set_camera_roi(x_min, y_min, width, height)
    except:
        pass

# --- Boucles d'update ---
update_image()
update_reference_image()

root.mainloop()
