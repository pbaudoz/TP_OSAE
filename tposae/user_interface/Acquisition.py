# acquisition.py
import os
import numpy as np
from catkit2.testbed import TestbedProxy

# === PARAMÈTRES ===
HOST = "127.0.0.1"
PORT = 1234
NUM_EXPOSURES = 1

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
    Elle demande à la caméra du télescope de prendre une nouvelle photo des spots lumineux, cette image est récupérée directment en mémoire vive par le programme.
    """
    camera = _connect_to_camera()
    # print("📸 Capture d'une image en direct...") # Commenté pour éviter trop de messages en temps réel
    try:
        # take_raw_exposures renvoie un générateur, next() pour obtenir la première image
        image = next(camera.take_raw_exposures(NUM_EXPOSURES))
        # print("✔️ Image capturée.") # Commenté pour éviter trop de messages en temps réel
        return image
    except Exception as e:
        print(f"❌ Erreur lors de la capture d'image : {e}")
        return None


def process_and_save_single_image(img, save_path, threshold=0.1):
    """
    Fonction utilitaire pour traiter et sauvegarder une image spécifique.
    Peut être appelée indépendamment.
    """
    img_processed = np.where(img > threshold, img, 0)
    np.save(save_path, img_processed)
    print(f"💾 Image traitée et enregistrée : {save_path}")
    return img_processed


if __name__ == "__main__":
    # Ce bloc s'exécute uniquement si acquisition.py est lancé directement
    save_dir_local = r"C:\msys64\home\Master2-OSAE\cam"
    os.makedirs(save_dir_local, exist_ok=True)

    live_image = get_live_image()
    if live_image is not None:
        processed_image = process_and_save_single_image(live_image, os.path.join(save_dir_local, "image_0.npy"))

        # Affichage rapide si nécessaire pour ce fichier uniquement
        import matplotlib.pyplot as plt

        plt.imshow(processed_image, cmap='gray')
        plt.title("Image traitée (depuis acquisition.py)")
        plt.colorbar()
        plt.show()