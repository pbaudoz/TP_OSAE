import numpy as np
from catkit2.testbed import TestbedProxy
import cv2

" I. CONNEXION A LA CAMERA "

# Paramètres propres à l'ordinateur et la caméra
HOST = "127.0.0.1"
PORT = 2345
NUM_EXPOSURES = 1
IMAGE_PATH = "image_reference.png"

# Initialisation des variables permettant la connexion à la caméra et au miroir
_tb = None
_camera = None

"F Connexion à la caméra si elle n'est pas déjà établie"
def _connect_to_camera():
    global _tb, _camera

    if _tb is None: # vérif si la connexion à la caméra n'est pas établie
        _tb = TestbedProxy(HOST, PORT)
        _camera = _tb.detector
        print("✅ Connexion au banc de test établie.")

    return _camera

"F Normalisation de l'exposition et fenêtrage de l'image dans la zone d'intérêt (ROI)"
def get_live_image():
    camera = _connect_to_camera() # connexion à la caméra

    try:
        raw_expo = next(camera.take_raw_exposures(NUM_EXPOSURES)) # Valeurs brutes de l'exposition en sortie de caméra

        max_val = np.max(raw_expo) 
        min_val = np.min(raw_expo)

        " Normalisation : boucle nécessaire pour pas diviser par 0 "
        if max_val > min_val: 
            image = (raw_expo - min_val) / (max_val - min_val) * 255

        else : 
            image = np.zeros_like(raw_expo)

        image = np.clip(image, 0, 255).astype(np.uint8) # Conversion en 8 bits

        " ROI : Fenêtre d'observation établie sur la zone d'intérêt (ROI) donnée par la caméra (si elle existe) "
        if hasattr(camera, "roi") and camera.roi is not None:
            x, y, w, h = camera.roi
            image = image[y:y+h, x:x+w] # Fenêtrage

        return image, max_val

    except Exception as e:
        print(f"❌ Erreur lors de la capture d'image : {e}")
        return None

"F Moyennage de n (par défaut = 500) images puis sauvegarde dans image_reference.png"
def capture_and_save_image(num_acquisitions=500):
    reset_camera_roi() # Image plein cadre
    
    " Initialisation "
    total_image_sum = None
    successful_captures = 0
    
    print(f"🔄 Début de la capture de {num_acquisitions} images pour la moyenne...")

    " Boucle d'Acquisition "
    for i in range(num_acquisitions):
        image, max_val = get_live_image()
        
        if image is not None:
            image_np = np.asarray(image)
            current_image_float = image_np.astype(np.float64) # Conversion de l'image en float
            
            if total_image_sum is None:
                total_image_sum = current_image_float # Initialisation de la somme
            else:
                total_image_sum += current_image_float
            
            successful_captures += 1
        else:
            print(f"⚠️ Avertissement : Acquisition {i+1} échouée. Tentative suivante...")
            # On pourrait ajouter ici un mécanisme de pause ou de réessai si nécessaire
    
    " Moyennage des images "
    if successful_captures > 0:
        average_image_float = total_image_sum / successful_captures
        average_image_uint8 = np.clip(average_image_float, 0, 255).astype(np.uint8) # Clip pour garder le format des images (entre 0 et 255) + Conversion en 8 bits
        
        " Sauvegarde de l'image "
        cv2.imwrite(IMAGE_PATH, average_image_uint8)
        print(f"✅ Moyenne de {successful_captures} images calculée et sauvegardée sous {IMAGE_PATH}")

    else:
        print("❌ Aucune image capturée avec succès. Impossible de calculer la moyenne.")

"F Affiche la caméra en temps réel + Eteint si utilisateur quitte l'interface "
def show_live():
    """Affiche la caméra en temps réel avec possibilité de fermer avec la croix."""
    while True: # Boucle infinie
        image, max_val = get_live_image()

        " Affiche la caméra "
        if image is not None:
            cv2.imshow("Camera Live", image)
        else:
            print("❌ Aucune image capturée.")
            break
        
        " Eteint si l'utilisateur a appuyé sur la touche Échap ou a fermé la fenêtre "
        key = cv2.waitKey(1) & 0xFF  # Récupère le code de la touche
        if key == 27:  # Touche Échap
            break
        if cv2.getWindowProperty("Camera Live", cv2.WND_PROP_VISIBLE) < 1:  # Fenêtre fermée par la croix
            break
    
    cv2.destroyAllWindows()

"C Activation de l'affchage en temps réel "
if __name__ == "__main__":
    show_live()

"F Fenêtrage de l'image sur une zone d'intérêt définie (ROI) "
def set_camera_roi(x, y, width, height):
    global _camera

    if _camera is None:
        _connect_to_camera()

    _camera.roi = (x, y, width, height)
    print(f"✅ ROI défini : x={x}, y={y}, w={width}, h={height}")

"F Réinitialisation de ROI : Plein écran "
def reset_camera_roi():
    global _camera

    if _camera is None:
        _connect_to_camera()

    _camera.roi = None 
    print("🔄 ROI réinitialisé : capture en pleine résolution.")

"F Etablit un temps d'exposition de la caméra "
def set_camera_exposure(exposure_us):
    global _camera

    if _camera is None:
        _connect_to_camera()

    try:
        _camera.exposure_time = exposure_us
        print(f"✅ Temps d'exposition réglé sur {exposure_us} µs.")
    except Exception as e:
        print(f"❌ Impossible de régler l'exposition : {e}")
