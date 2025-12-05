import os
import numpy as np
from catkit2.testbed import TestbedProxy
import cv2
from Reference import detect_spots

# === PARAMÈTRES ===
HOST = "127.0.0.1"
PORT = 2345
NUM_EXPOSURES = 1
IMAGE_PATH = "image_reference.png"

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
    
def capture_and_save_image(num_acquisitions=500):
    """
    Capture et moyenne un nombre spécifié d'images avant de sauvegarder le résultat.
    
    :param num_acquisitions: Le nombre d'images à capturer et moyenner.
    """
    reset_camera_roi()
    
    # 1. Initialisation
    total_image_sum = None
    successful_captures = 0
    
    print(f"🔄 Début de la capture de {num_acquisitions} images pour la moyenne...")

    # 2. Boucle d'Acquisition
    for i in range(num_acquisitions):
        image = get_live_image()
        
        if image is not None:
            # Assurez-vous que l'image est un tableau NumPy
            image_np = np.asarray(image)
            
            # Convertir l'image en float64 pour éviter l'overflow lors de la somme
            # et pour permettre la division finale non entière.
            current_image_float = image_np.astype(np.float64) 
            
            if total_image_sum is None:
                # Initialise la somme avec la première image
                total_image_sum = current_image_float
            else:
                # Ajoute l'image actuelle à la somme totale
                total_image_sum += current_image_float
            
            successful_captures += 1
        else:
            print(f"⚠️ Avertissement : Acquisition {i+1} échouée. Tentative suivante...")
            # On pourrait ajouter ici un mécanisme de pause ou de réessai si nécessaire

    # ---
    
    # 3. Calcul de la Moyenne et Sauvegarde
    if successful_captures > 0:
        # Calcul de la moyenne par division par le nombre d'acquisitions réussies
        average_image_float = total_image_sum / successful_captures
        
        # Convertir le résultat en type entier non signé 8 bits (le format standard des images)
        # On utilise np.clip pour s'assurer que les valeurs restent entre 0 et 255.
        average_image_uint8 = np.clip(average_image_float, 0, 255).astype(np.uint8)
        
        # Sauvegarde l'image moyennée
        cv2.imwrite(IMAGE_PATH, average_image_uint8)
        print(f"✅ Moyenne de {successful_captures} images calculée et sauvegardée sous {IMAGE_PATH}")
    else:
        print("❌ Aucune image capturée avec succès. Impossible de calculer la moyenne.")

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
