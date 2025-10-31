import os
import numpy as np
from catkit2.testbed import TestbedProxy
import cv2

HOST = "127.0.0.1"
PORT = 2345
NUM_EXPOSURES = 1
IMAGE_PATH = "image_reference.jpg"

_tb = None
_camera = None

def _connect_to_camera():
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
        image = np.clip(image,0,255).astype(np.uint8)
        if hasattr(camera, "roi"):
            x, y, w, h = camera.roi
            image = image[y:y+h, x:x+w]
        return image
    except Exception as e:
        print(f"❌ Erreur capture image : {e}")
        return None

def capture_and_save_image():
    image = get_live_image()
    if image is not None:
        cv2.imwrite(IMAGE_PATH, image)
        print(f"✅ Image sauvegardée sous {IMAGE_PATH}")
    else:
        print("❌ Aucune image capturée.")

def set_camera_roi(x, y, width, height):
    global _camera
    if _camera is None:
        _connect_to_camera()
    _camera.roi = (x,y,width,height)
    print(f"✅ ROI défini : x={x},y={y},w={width},h={height}")

def set_camera_exposure(exposure_us):
    global _camera
    if _camera is None:
        _connect_to_camera()
    try:
        _camera.exposure_time = exposure_us
        print(f"✅ Temps d'exposition réglé sur {exposure_us/1e6:.6f} s")
    except Exception as e:
        print(f"❌ Impossible de régler l'exposition : {e}")
