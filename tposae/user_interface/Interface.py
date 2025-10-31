import tkinter as tk
from tkinter import Label, Button, Entry
from PIL import Image, ImageTk
import cv2
import numpy as np
from visualisation import get_live_image, set_camera_roi, set_camera_exposure, toggle_grid_display, grid_visible
from Reference import process_and_save_images
from visualisation import capture_and_save_image

# --- Lecture des coordonnées depuis le fichier ---
def assign_coordinates_from_file(filename='reference.txt'):
    coords = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if ":" in line:
                    key, val = line.split(":")
                    coords[key.strip()] = val.strip()
        print(f"✅ Coordonnées/Grille chargées depuis {filename}")
        return coords
    except Exception as e:
        print(f"❌ Erreur lors de l'assignation des coordonnées : {e}")
        return None


# --- Interface principale ---
def show_live_and_reference_image():
    coords = assign_coordinates_from_file('reference.txt')
    if coords and all(k in coords for k in ['x_min', 'x_max', 'y_min', 'y_max']):
        x_min, x_max = int(coords['x_min']), int(coords['x_max'])
        y_min, y_max = int(coords['y_min']), int(coords['y_max'])
        width = x_max - x_min
        height = y_max - y_min
        set_camera_roi(x_min, y_min, width, height)

    def update_image():
        """Met à jour l'image en temps réel avec ou sans grille."""
        image = get_live_image()
        if image is not None:
            if grid_visible:
                image = draw_grid_on_image(image)
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img_pil)
            live_video_label.config(image=img_tk)
            live_video_label.image = img_tk
        root.after(50, update_image)

    def update_reference_image():
        """Met à jour l'image de référence dans le Tkinter."""
        try:
            reference_image = cv2.imread('image_reference_centre.jpg')
            if reference_image is not None:
                if grid_visible:
                    reference_image = draw_grid_on_image(reference_image)
                img_pil = Image.fromarray(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(img_pil)
                reference_image_label.config(image=img_tk)
                reference_image_label.image = img_tk
        except Exception as e:
            print(f"Erreur lors de l'affichage de l'image de référence : {e}")
        root.after(1000, update_reference_image)

    def acquire_new_reference_image():
        """Acquérir et traiter une nouvelle image de référence."""
        capture_and_save_image()
        process_and_save_images('image_reference.jpg')
        update_reference_image()

    def update_exposure():
        """Met à jour le temps d'exposition."""
        try:
            exposure_s = float(exposure_entry.get())
            exposure_us = exposure_s * 1e6
            set_camera_exposure(exposure_us)
            print(f"✅ Exposition réglée sur {exposure_s:.3f} s")
        except Exception as e:
            print(f"❌ Erreur lors du réglage de l'exposition : {e}")

    def toggle_grid():
        toggle_grid_display()
        print("✅ Grille affichée" if grid_visible else "🚫 Grille masquée")

    # --- Interface graphique ---
    root = tk.Tk()
    root.title("Caméra en Temps Réel et Image de Référence")

    live_video_frame = tk.Frame(root)
    live_video_frame.pack(side=tk.LEFT, padx=10, pady=10)
    live_video_label = Label(live_video_frame)
    live_video_label.pack()

    reference_image_frame = tk.Frame(root)
    reference_image_frame.pack(side=tk.RIGHT, padx=10, pady=10)
    reference_image_label = Label(reference_image_frame)
    reference_image_label.pack()

    acquire_button = Button(root, text="Acquérir une nouvelle image de référence", command=acquire_new_reference_image)
    acquire_button.pack(pady=10)

    exposure_label = Label(root, text="Exposition (s) :")
    exposure_label.pack()
    exposure_entry = Entry(root)
    exposure_entry.insert(0, "0.02")
    exposure_entry.pack()
    exposure_button = Button(root, text="Appliquer", command=update_exposure)
    exposure_button.pack(pady=5)

    grid_button = Button(root, text="Afficher / Masquer la grille", command=toggle_grid)
    grid_button.pack(pady=10)

    update_image()
    update_reference_image()
    root.mainloop()


# --- Fonction de dessin de grille (pour l'affichage) ---
def draw_grid_on_image(image, filename='reference.txt'):
    """Dessine la grille sur une image à partir du fichier de référence."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        grid_lines = [l for l in lines if l.startswith("grid_line")]
        for line in grid_lines:
            _, vals = line.strip().split(":")
            x1, y1, x2, y2 = map(int, vals.split(","))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return image
    except Exception as e:
        print(f"Erreur dessin grille : {e}")
        return image


if __name__ == "__main__":
    show_live_and_reference_image()
