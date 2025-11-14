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
    camera = _connect_to_camera()
    try:
        image = next(camera.take_raw_exposures(NUM_EXPOSURES))
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Découpe du ROI si défini et non None
        if hasattr(camera, "roi") and camera.roi is not None:
            x, y, w, h = camera.roi
            image = image[y:y+h, x:x+w]

        return image

    except Exception as e:
        print(f"❌ Erreur lors de la capture d'image : {e}")
        return None
    
def capture_and_save_image():
    """Capture une image unique et l'enregistre, écrasant l'ancienne."""
    reset_camera_roi()
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

def set_camera_roi(x, y, width, height):
    """
    Définit une région d'intérêt (ROI) dans l'image capturée.
    ⚠️ La caméra du banc de test ne gère pas forcément ça matériellement,
    donc on découpe l'image après capture.
    """
    global _camera
    if _camera is None:
        _connect_to_camera()

    # On ne peut pas vraiment configurer le ROI matériellement,
    # donc on mémorise la zone pour la découper au besoin.
    _camera.roi = (x, y, width, height)
    print(f"✅ ROI défini : x={x}, y={y}, w={width}, h={height}")

def reset_camera_roi():
    """Réinitialise le ROI pour capturer l'image complète."""
    global _camera
    if _camera is None:
        _connect_to_camera()

    _camera.roi = None   # <--- ROI désactivé
    print("🔄 ROI réinitialisé : capture en pleine résolution.")


def set_camera_exposure(exposure_us):
    """
    Définit le temps d'exposition de la caméra, en microsecondes.
    """
    global _camera
    if _camera is None:
        _connect_to_camera()

    try:
        _camera.exposure_time = exposure_us
        print(f"✅ Temps d'exposition réglé sur {exposure_us / 1e6:.3f} s")
    except Exception as e:
        print(f"❌ Impossible de régler l'exposition : {e}")
