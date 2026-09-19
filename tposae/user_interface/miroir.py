# -*- coding: utf-8 -*-
from ctypes import *
import time
import numpy as np
import threading
import re
from Reference import detect_spots, read_grid_from_reference, assign_coordinates_from_file
from visualisation import get_live_image
from Vecteur_spots import compute_cells_from_grid_ref, read_grid_cells_from_reference

# === VARIABLES GLOBALES ===
lib = None
libX = None
instrumentHandle = None
segmentCount = None
tiltCount = None


def initialisation_miroir():
    global lib, libX, instrumentHandle, segmentCount, tiltCount

    print("=== Initialisation du miroir DMP40 ===")

    lib = cdll.LoadLibrary("C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLDFM_64.dll")
    libX = cdll.LoadLibrary("C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLDFMX_64.dll")

    instrumentHandle = c_ulong()
    IDQuery = True
    resetDevice = False
    resource = c_char_p(b"")
    deviceCount = c_int()

    lib.TLDFM_get_device_count(instrumentHandle, byref(deviceCount))
    if deviceCount.value < 1:
        print("❌ Aucun miroir DMP40 détecté.")
        return False

    print(f"➡️ {deviceCount.value} miroir(s) détecté(s).")

    lib.TLDFM_get_device_information(instrumentHandle, 0, 0, 0, 0, 0, resource)

    if libX.TLDFMX_init(resource.value, IDQuery, resetDevice, byref(instrumentHandle)) != 0:
        print("❌ Erreur d'initialisation du miroir.")
        return False

    print("✅ Connexion au miroir établie.")

    segmentCount = c_uint32()
    tiltCount = c_uint32()

    lib.TLDFM_get_segment_count(instrumentHandle, byref(segmentCount))
    lib.TLDFM_get_tilt_count(instrumentHandle, byref(tiltCount))

    print(f"➡️ Segments : {segmentCount.value}, Tilt arms : {tiltCount.value}")
    print("Initialisation terminée.\n")

    return True



def relax_miroir():
    global lib, libX, instrumentHandle, segmentCount, tiltCount

    if lib is None or libX is None or instrumentHandle is None:
        print("❌ Le miroir n'est pas initialisé.")
        return

    print("=== Relaxation du miroir DMP40 ===")

    devicePart = c_uint32(2)
    isFirstStep = c_bool(True)
    reload = c_bool(False)

    segCount = segmentCount.value
    tiltC = tiltCount.value

    relaxPatternMirror = (c_double * segCount)()
    relaxPatternArms = (c_double * tiltC)()

    remainingSteps = c_int32()
    counter = 1

    libX.TLDFMX_relax(
        instrumentHandle, devicePart, isFirstStep, reload,
        relaxPatternMirror, relaxPatternArms, byref(remainingSteps)
    )

    lib.TLDFM_set_segment_voltages(instrumentHandle, relaxPatternMirror)
    lib.TLDFM_set_tilt_voltages(instrumentHandle, relaxPatternArms)
    print("Relax step:", counter)
    counter += 1

    isFirstStep = c_bool(False)

    while remainingSteps.value > 0:
        libX.TLDFMX_relax(
            instrumentHandle, devicePart, isFirstStep, reload,
            relaxPatternMirror, relaxPatternArms, byref(remainingSteps)
        )
        lib.TLDFM_set_segment_voltages(instrumentHandle, relaxPatternMirror)
        lib.TLDFM_set_tilt_voltages(instrumentHandle, relaxPatternArms)
        print("Relax step:", counter)
        counter += 1
    print("focus :", np.array(relaxPatternMirror))
    print("tilt :", np.array(relaxPatternArms))
    

    print("✅ Relaxation complète.\n")


#fonction pour activer un unique piston
def activation_piston(numero_piston, tension, voltage_piston = np.ones(40)*0):
    voltage_piston[numero_piston] = tension
    type_c_voltage_piston = c_double * 40
    c_pattern = type_c_voltage_piston(*voltage_piston)
    lib.TLDFM_set_segment_voltages(instrumentHandle, c_pattern)
    time.sleep(1)  #on le laisse mais à voir
    return voltage_piston


def set_all_pistons(voltage = 70):
    """
    Applique la même tension à tous les pistons du miroir DMP40.
    
    Args:
        voltage (float): La tension à appliquer à chaque piston [V].
    """
    global lib, instrumentHandle, segmentCount

    if lib is None or instrumentHandle is None or segmentCount is None:
        print("❌ Le miroir n'est pas initialisé.")
        return

    # Crée un tableau de la taille des segments rempli de la tension désirée
    pattern = [voltage] * segmentCount.value

    # Convertit en type compatible ctypes
    type_c_pattern = c_double * segmentCount.value
    c_pattern = type_c_pattern(*pattern)

    # Applique la tension à tous les pistons
    lib.TLDFM_set_segment_voltages(instrumentHandle, c_pattern)
    print(f"✅ Tous les pistons mis à {voltage} V.")

    tiltPattern = (c_double * 3)()  # pour 3 pistons tilt
    tiltPattern[0] = 110  # ajustement piston 1
    tiltPattern[1] = 100 # ajustement piston 2
    tiltPattern[2] = 100  # ajustement piston 3
    lib.TLDFM_set_tilt_voltages(instrumentHandle, tiltPattern)

def compute_interaction_vector(current_cells, ref_cells):
    """
    Génère un vecteur de signal (dx1, dx2..., dy1, dy2...) de taille constante
    pour le calcul de la matrice d'interaction.
    """
    dx_vector = []
    dy_vector = []

    # On transforme les cellules actuelles en dictionnaire pour un accès rapide
    current_dict = {(c['cx'], c['cy']): c for c in current_cells}

    # On boucle sur la RÉFÉRENCE (qui est notre structure fixe)
    for ref in ref_cells:
        # On ne s'intéresse qu'aux cases qui DOIVENT avoir un spot
        if ref['has_spot']:
            key = (ref['cx'], ref['cy'])
            
            dx = 0.0
            dy = 0.0
            
            # Si la cellule existe dans l'image actuelle ET qu'un spot y est vu
            if key in current_dict and current_dict[key]['has_spot']:
                dx = current_dict[key]['xspot'] - ref['xspot']
                dy = current_dict[key]['yspot'] - ref['yspot']
            
            dx_vector.append(dx)
            dy_vector.append(dy)

    # On concatène pour avoir [dx1, dx2... dxN, dy1, dy2... dyN]
    return np.array(dx_vector + dy_vector)

def matrice_interaction():
    def worker():
        global lib, instrumentHandle, segmentCount
        if lib is None or instrumentHandle is None or segmentCount is None:
            print("❌ Le miroir n'est pas initialisé.")
            return

        # --- PRÉPARATION ---
        # 1. Lire la référence UNE SEULE FOIS avant la boucle
        ref_cells = read_grid_cells_from_reference('reference.txt')
        vertical_lines, horizontal_lines = read_grid_from_reference('reference.txt')
        coords = assign_coordinates_from_file('reference.txt')
        x_offset = coords.get('x_min', 0)
        y_offset = coords.get('y_min', 0)
        
        pattern = np.ones(segmentCount.value) * 70
        MI = []

        # --- BOUCLE DE CALIBRATION ---
        for numero_piston in range(segmentCount.value):
            # Activer le piston
            pattern[numero_piston] = 180
            type_c_pattern = c_double * segmentCount.value
            c_pattern = type_c_pattern(*pattern)
            lib.TLDFM_set_segment_voltages(instrumentHandle, c_pattern)
            
            time.sleep(1) 
            
            # Capturer l'image actuelle (Attention: assure-toi que get_live_image() est à jour)
            frame = get_live_image() 
            centers_roi, _ = detect_spots(frame)
            
            # Recalculer les coordonnées full frame
            centers_full_ref = [(cx + x_offset, cy + y_offset) for cx, cy in centers_roi]
            
            # Calculer les cellules de l'image actuelle
            current_cells = compute_cells_from_grid_ref(centers_full_ref, vertical_lines, horizontal_lines)
            
            # GÉNÉRATION DU VECTEUR ROBUSTE (Taille fixe, remplit de 0.0 si spot perdu)
            vector_signal = compute_interaction_vector(current_cells, ref_cells)
            MI.append(vector_signal)

            # Revenir à l'état repos
            pattern[numero_piston] = 70
            lib.TLDFM_set_segment_voltages(instrumentHandle, type_c_pattern(*pattern))
            time.sleep(0.5)
            
        print("✅ Acquisition MI terminée. Calcul de la pseudo-inverse...")

        # --- CALCUL ET SAUVEGARDE ---
        matrice_MI = np.array(MI)
        # Utilise un rcond un peu plus élevé (1e-3 ou 1e-2) pour filtrer le bruit
        MC = np.linalg.pinv(matrice_MI, rcond=1e-2)

        header = "\n#BEGIN matrice de controle\n"
        footer = "#END matrice de controle\n"
        matrix_text = ""
        for row in MC:
            matrix_text += ",".join(f"{val:.8f}" for val in row) + "\n"
        
        new_section = header + matrix_text + footer

        try:
            with open('reference.txt', 'r') as f:
                content = f.read()

            import re
            pattern_regex = r"#BEGIN matrice de controle.*?#END matrice de controle\n?"
            
            if re.search(pattern_regex, content, re.DOTALL):
                new_content = re.sub(pattern_regex, new_section, content, flags=re.DOTALL)
            else:
                new_content = content.strip() + "\n" + new_section

            with open('reference.txt', 'w') as f:
                f.write(new_content)
                
            print("💾 Section 'Matrice de controle' mise à jour avec succès.")
        
            
        except Exception as e:
            print(f"❌ Erreur lors de l'écriture : {e}")

        header = "\n#BEGIN MI\n"
        footer = "#END MI\n"
        matrix_text = ""
        for row in MI:
            matrix_text += ",".join(f"{val:.8f}" for val in row) + "\n"
        
        new_section = header + matrix_text + footer

        try:
            with open('reference.txt', 'r') as f:
                content = f.read()

            import re
            pattern_regex = r"#BEGIN MI.*?#END MI\n?"
            
            if re.search(pattern_regex, content, re.DOTALL):
                new_content = re.sub(pattern_regex, new_section, content, flags=re.DOTALL)
            else:
                new_content = content.strip() + "\n" + new_section

            with open('reference.txt', 'w') as f:
                f.write(new_content)
                
            print("💾 Section 'MI' mise à jour avec succès.")
        
            
        except Exception as e:
            print(f"❌ Erreur lors de l'écriture : {e}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
