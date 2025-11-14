import tkinter as tk
from tkinter import Label, Button, Entry
from PIL import Image, ImageTk
import cv2
import numpy as np
from visualisation import get_live_image, set_camera_roi, set_camera_exposure
from visualisation import capture_and_save_image
from Reference import process_and_save_images, assign_coordinates_from_file

def show_live_and_reference_image():
    """Affiche la caméra en temps réel et l'image de référence dans une fenêtre Tkinter."""
    
    # Lire les coordonnées depuis le fichier (ROI initial)
    coords = assign_coordinates_from_file('reference.txt')
    
    if coords:
        x_min, x_max = coords['x_min'], coords['x_max']
        y_min, y_max = coords['y_min'], coords['y_max']
        width = x_max - x_min
        height = y_max - y_min
        set_camera_roi(x_min, y_min, width, height)
    else:
        print("❌ Aucune coordonnée lue depuis le fichier 'reference.txt'")

    def update_image():
        """Met à jour l'image en temps réel de la caméra."""
        image = get_live_image()
        if image is not None:
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img_pil)
            live_video_label.config(image=img_tk)
            live_video_label.image = img_tk
        root.after(10, update_image)
    
    def update_reference_image():
        """Met à jour l'image de référence à partir du fichier 'image_reference_centre.jpg'."""
        try:
            reference_image = cv2.imread('image_reference_centre.jpg')
            if reference_image is not None:
                img_pil = Image.fromarray(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(img_pil)
                reference_image_label.config(image=img_tk)
                reference_image_label.image = img_tk
        except Exception as e:
            print(f"Erreur lors de l'affichage de l'image de référence : {e}")
        root.after(1000, update_reference_image)

    def acquire_new_reference_image():
        """Capture une nouvelle image de référence et met à jour le ROI."""
        print("🔵 Acquisition d’une nouvelle image de référence...")
        capture_and_save_image()
        process_and_save_images('image_reference.jpg')
        coords = assign_coordinates_from_file('reference.txt')
        if coords:
            x_min = coords['x_min']
            x_max = coords['x_max']
            y_min = coords['y_min']
            y_max = coords['y_max']
            width = x_max - x_min
            height = y_max - y_min
            set_camera_roi(x_min, y_min, width, height)
            print(f"🎯 Nouveau ROI appliqué : {coords}")
        update_reference_image()

    def update_exposure():
        """Met à jour l'exposition de la caméra."""
        try:
            exposure_s = float(exposure_entry.get())
            exposure_us = exposure_s * 1e6
            set_camera_exposure(exposure_us)
            print(f"✅ Exposition réglée sur {exposure_s:.3f} s")
        except Exception as e:
            print(f"❌ Erreur lors du réglage de l'exposition : {e}")

    def apply_zoom_roi():
        """Applique le ROI de zoom défini dans le fichier reference.txt."""
        coords = assign_coordinates_from_file('reference.txt')
        if coords:
            x_min, x_max = coords['x_min'], coords['x_max']
            y_min, y_max = coords['y_min'], coords['y_max']
            width = x_max - x_min
            height = y_max - y_min
            set_camera_roi(x_min, y_min, width, height)
            print(f"🎯 ROI de Zoom appliqué : {coords}")
        else:
            print("❌ Aucune coordonnée lue depuis le fichier 'reference.txt'")

    # ============================================
    #              Interface Graphique
    # ============================================
    
    root = tk.Tk()
    root.title("Caméra en Temps Réel et Image de Référence")
    
    # --- Section vidéo live ---
    live_video_frame = tk.Frame(root)
    live_video_frame.pack(side=tk.LEFT, padx=10, pady=10)
    live_video_label = Label(live_video_frame)
    live_video_label.pack()

    # --- Section image de référence ---
    reference_image_frame = tk.Frame(root)
    reference_image_frame.pack(side=tk.RIGHT, padx=10, pady=10)
    reference_image_label = Label(reference_image_frame)
    reference_image_label.pack()

    # --- Bouton acquisition référence ---
    acquire_button = Button(root, text="Acquérir une nouvelle image de référence", command=acquire_new_reference_image)
    acquire_button.pack(pady=10)

    # --- Section Exposition ---
    exposure_label = Label(root, text="Exposition (s) :")
    exposure_label.pack()
    exposure_entry = Entry(root)
    exposure_entry.insert(0, "0.02")  # Valeur par défaut
    exposure_entry.pack()
    exposure_button = Button(root, text="Appliquer", command=update_exposure)
    exposure_button.pack(pady=5)

    # --- Bouton Zoom ---
    zoom_button = Button(root, text="Zoom (ROI)", command=apply_zoom_roi)
    zoom_button.pack(pady=10)

    # Lancer les mises à jour
    update_image()
    update_reference_image()
    
    # Démarrer l'interface graphique
    root.mainloop()

if __name__ == "__main__":
    show_live_and_reference_image()
