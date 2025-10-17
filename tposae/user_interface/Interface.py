import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from visualisation import get_live_image
from Reference import process_and_save_images  # Assurez-vous que cette fonction est bien définie
from visualisation import capture_and_save_image  # Assurez-vous que cette fonction est définie

# Fonction pour assigner les coordonnées x_min, x_max, y_min, y_max depuis le fichier
def assign_coordinates_from_file(filename='reference.txt'):
    """Assigner les coordonnées xmin, xmax, ymin, ymax directement à partir d'un fichier texte."""
    global x_min, x_max, y_min, y_max
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('x_min'):
                    x_min = int(line.split(": ")[1])
                elif line.startswith('x_max'):
                    x_max = int(line.split(": ")[1])
                elif line.startswith('y_min'):
                    y_min = int(line.split(": ")[1])
                elif line.startswith('y_max'):
                    y_max = int(line.split(": ")[1])

        print(f"✅ Coordonnées assignées : x_min = {x_min}, x_max = {x_max}, y_min = {y_min}, y_max = {y_max}")
    except Exception as e:
        print(f"❌ Erreur lors de l'assignation des coordonnées : {e}")

# Fonction pour afficher et traiter l'image
def show_live_and_reference_image():
    """Affiche la caméra en temps réel et l'image zoomée avec les centres lumineux dans une fenêtre Tkinter."""

    # Lire les coordonnées depuis le fichier
    assign_coordinates_from_file('reference.txt')

    def update_image():
        """Met à jour l'image affichée dans la fenêtre Tkinter pour la caméra en temps réel."""
        image = get_live_image()  # Capture de l'image en temps réel depuis la caméra
        if image is not None:
            # Appliquer un zoom sur l'image en temps réel en utilisant les coordonnées lues
            height, width = image.shape[:2]
            
            # Limiter les coordonnées pour qu'elles ne dépassent pas les dimensions de l'image
            zoom_x_min = max(0, min(x_min, width - 1))
            zoom_x_max = max(0, min(x_max, width))
            zoom_y_min = max(0, min(y_min, height - 1))
            zoom_y_max = max(0, min(y_max, height))

            # Découper l'image en fonction des coordonnées
            zoomed_image = image[zoom_y_min:zoom_y_max, zoom_x_min:zoom_x_max]

            # Convertir l'image NumPy en image PIL pour Tkinter
            img_pil = Image.fromarray(cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img_pil)

            # Mettre à jour l'image dans le label Tkinter
            live_video_label.config(image=img_tk)
            live_video_label.image = img_tk  # Référence pour éviter la suppression par le GC
        else:
            print("❌ Aucune image capturée.")
        
        # Continuer à mettre à jour toutes les 10ms
        root.after(10, update_image)

    def update_reference_image():
        """Met à jour l'image de référence dans le Tkinter."""
        try:
            # Charger l'image zoomée avec les pixels rouges marqués
            reference_image = cv2.imread('image_reference_centre.jpg')  # Assure-toi que cette image existe
            if reference_image is not None:
                # Convertir l'image NumPy en image PIL pour Tkinter
                img_pil = Image.fromarray(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(img_pil)

                # Mettre à jour l'image dans le label Tkinter
                reference_image_label.config(image=img_tk)
                reference_image_label.image = img_tk  # Référence pour éviter la suppression par le GC
            else:
                print("❌ Impossible de charger l'image de référence.")
        except Exception as e:
            print(f"Erreur lors de l'affichage de l'image de référence : {e}")
        
        # Continuer à mettre à jour toutes les 1000ms (1 seconde)
        root.after(1000, update_reference_image)

    def acquire_new_reference_image():
        """Acquérir une nouvelle image de référence, traiter et mettre à jour l'image de référence."""
        # Appeler la fonction pour capturer et sauvegarder une nouvelle image de référence
        capture_and_save_image()  # Cette fonction doit capturer et sauvegarder 'image_reference.jpg'
        
        # Appeler la fonction pour traiter l'image et générer l'image de référence avec les centres lumineux
        process_and_save_images('image_reference.jpg')  # Cette fonction doit traiter l'image et sauvegarder 'image_reference_centre.jpg'

        # Mettre à jour l'image de référence affichée
        update_reference_image()

    # Créer la fenêtre Tkinter
    root = tk.Tk()
    root.title("Caméra en Temps Réel et Image de Référence")

    # Créer un frame pour la caméra en temps réel
    live_video_frame = tk.Frame(root)
    live_video_frame.pack(side=tk.LEFT, padx=10, pady=10)

    # Créer un label pour afficher l'image de la caméra en temps réel
    live_video_label = Label(live_video_frame)
    live_video_label.pack()

    # Créer un frame pour l'image de référence
    reference_image_frame = tk.Frame(root)
    reference_image_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    # Créer un label pour afficher l'image de référence (zoomée avec les pixels rouges)
    reference_image_label = Label(reference_image_frame)
    reference_image_label.pack()

    # Créer un bouton pour acquérir une nouvelle image de référence
    acquire_button = Button(root, text="Acquérir une nouvelle image de référence", command=acquire_new_reference_image)
    acquire_button.pack(pady=10)

    # Lancer la mise à jour des images
    update_image()  # Lancer la mise à jour de l'image en temps réel
    update_reference_image()  # Lancer la mise à jour de l'image de référence

    # Démarrer la boucle Tkinter
    root.mainloop()

if __name__ == "__main__":
    show_live_and_reference_image()
