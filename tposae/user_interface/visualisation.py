import os
import numpy as np
from catkit2.testbed import TestbedProxy
import cv2

# === PARAMÈTRES ===
HOST = "127.0.0.1"
PORT = 2345
NUM_EXPOSURES = 1
IMAGE_PATH = "image_reference.jpg"

# Connexion globale pour éviter de se connecter/déconnecter à chaque image
_tb = None
_camera = None

def _connect_to_camera():
    """Établit la connexion à la caméra si elle n'est pas déjà établie."""
    global _tb, _camera
    if _tb is None:
        _tb = TestbedProxy(HOST, PORT)
        _camera = _tb.detector
        print("✅ Connexion au banc de test établie.")
    return _camera

def get_live_image():
    """
    Capture une image de la caméra et la renvoie sous forme de tableau NumPy.
    Gère la connexion à la caméra.
    """
    camera = _connect_to_camera()
    try:
        # take_raw_exposures renvoie un générateur, next() pour obtenir la première image
        image = next(camera.take_raw_exposures(NUM_EXPOSURES))
        
        # --- Normalisation de l'image --- 
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)  # Assure que l'image reste entre 0 et 255
        
        return image
    except Exception as e:
        print(f"❌ Erreur lors de la capture d'image : {e}")
        return None

def capture_and_save_image():
    """Capture une image unique et l'enregistre, écrasant l'ancienne."""
    image = get_live_image()
    if image is not None:
        # Sauvegarde l'image avec le nom spécifié (écrase la précédente)
        cv2.imwrite(IMAGE_PATH, image)
        print(f"✅ Image sauvegardée sous {IMAGE_PATH}")
    else:
        print("❌ Aucune image capturée pour sauvegarde.")

def show_live():
    """Affiche la caméra en temps réel avec possibilité de fermer avec la croix."""
    while True:
        image = get_live_image()
        if image is not None:
            cv2.imshow("Camera Live", image)
        else:
            print("❌ Aucune image capturée.")
            break
        
        # Vérifie si l'utilisateur a appuyé sur la touche Échap ou a fermé la fenêtre
        key = cv2.waitKey(1) & 0xFF  # Récupère le code de la touche
        if key == 27:  # Touche Échap
            break
        if cv2.getWindowProperty("Camera Live", cv2.WND_PROP_VISIBLE) < 1:  # Fenêtre fermée par la croix
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_live()
