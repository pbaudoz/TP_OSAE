"""
Commandes nécessaires pour démarrer l'acquisition des images via catkit2 sur une machine de l'observatoire :
Dans un terminal catkit2 : controller start server

A partir de ce moment là, on peut charger le fichier python (Acquisition.py)
Il ne faut "que" 4 commandes pour se connecter à la camera : 
from catkit2 import TestbedProxy
testbed = TestbedProxy("127.0.0.1", 1234)
camera = testbed.detector
#definir le nombres d'expositions voulues (num_exposures)
camera.take_raw_exposures(num_exposures)
#Boucle numpy pour enregistrer les images au format souhaiter
camera.end_acquisition()

Concernant le modèle de caméra : 
Actuellement au 5/01/2025 : 
 camera_id: "DEV_1AB22C03E846"
 device_name: Allied Alvium 1800 U-500m

Si changement : 
Nécessite de modifier le fichier services.yml
avec les variables : 
    camera_id
    device_name
Ces paramètres de la caméra sont accessible en branchant la caméra alvium sur un logiciel VimbaViewver

Attention,  au 5/01/2025, impossible de modiffier pour augmenter les valeurs des variables suivantes : width, height, offset_x, offset_y, 

En complément : 
print(dir(camera))
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_service_interfaces', '_testbed', 'command_names', 'config', 'data_stream_names', 'execute_command', 'get_data_stream', 'get_property', 'get_service_interface', 'heartbeat', 'id', 'interrupt', 'is_alive', 'is_running', 'property_names', 'register_service_interface', 'set_property', 'start', 'state', 'stop', 'take_exposures', 'take_raw_exposures', 'terminate', 'testbed']

print(dir(camera.images))

['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'buffer_handling_mode', 'copy', 'create', 'dtype', 'frame_rate', 'get', 'get_frame', 'get_latest_frame', 'get_next_frame', 'is_frame_available', 'newest_available_frame_id', 'num_frames_in_buffer', 'oldest_available_frame_id', 'open', 'owner_pid', 'request_new_frame', 'shape', 'stream_id', 'stream_name', 'submit_data', 'submit_frame', 'time_created', 'update_parameters', 'version', 'will_frame_be_available']

print(type(image), type(image1), type(im))
<class 'generator'> <class 'generator'> <class 'catkit2.catkit_bindings.DataFrame'>

Lien vers le github : https://github.com/spacetelescope/catkit2/tree/develop/docs

"""

from catkit2 import TestbedProxy
import numpy as np
from PIL import Image

# Connexion au banc de test
testbed = TestbedProxy("127.0.0.1", 1234)
camera = testbed.detector
print("Connexion au banc de test établie.")

# Nbre captures
num_exposures = 5  # Par exemple, 5 images

# Capture avec `take_raw_exposures`
print(f"Capture de {num_exposures} images...")
image_generator = camera.take_raw_exposures(num_exposures)

# Sauvegarde
for i, img in enumerate(image_generator):
    print(f"Image {i+1}/{num_exposures} capturée.")

    # Sauvegarde au format NumPy
    np.save(f"image_{i}.npy", img)
    print(f"Image {i} sauvegardée au format NumPy : 'image_{i}.npy'.")

    # Conversion PNG
    img_uint8 = (img - img.min()) / (img.max() - img.min()) * 255  # Normalisation entre 0 et 255
    img_uint8 = img_uint8.astype(np.uint8)  # uint8

    # Save
    img_png = Image.fromarray(img_uint8)
    img_png.save(f"image_{i}.png")
    print(f"Image {i} sauvegardée au format PNG : 'image_{i}.png'.")

print("Toutes les images ont été capturées et sauvegardées.")

camera.end_acquisition()


