# -*- coding: utf-8 -*-
from ctypes import *
import time
import numpy as np
import threading
from Reference import detect_spots, read_grid_from_reference, assign_coordinates_from_file
from visualisation import get_live_image
from Vecteur_spots import compute_cells_from_grid_ref, compare_grid_cells_and_compute_vectors

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


def matrice_interaction():
    """
    Version thread-safe de matrice_interaction() pour PyQt.
    Exécute la séquence de mise à 0V et retour à 70V sur chaque piston
    dans un thread séparé pour ne pas bloquer l'interface.
    """
    def worker():
        global lib, instrumentHandle, segmentCount
        if lib is None or instrumentHandle is None or segmentCount is None:
            print("❌ Le miroir n'est pas initialisé.")
            return

        pattern = np.ones(segmentCount.value) * 70
        frame = get_live_image()
        vertical_lines, horizontal_lines = read_grid_from_reference('reference.txt')
        coords = assign_coordinates_from_file('reference.txt')

        for numero_piston in range(segmentCount.value):
            pattern[numero_piston] = 10
            type_c_pattern = c_double * segmentCount.value
            c_pattern = type_c_pattern(*pattern)
            lib.TLDFM_set_segment_voltages(instrumentHandle, c_pattern)
            
            time.sleep(1)  # Pause d'une seconde pour chaque piston
            
            centers_roi, _ = detect_spots(frame)
                
            x_offset = coords.get('x_min', 0)
            y_offset = coords.get('y_min', 0)
                
            centers_full_ref = []
            for cx_roi, cy_roi in centers_roi:
                centers_full_ref.append((cx_roi + x_offset, cy_roi + y_offset))
            cells = compute_cells_from_grid_ref(centers_full_ref, vertical_lines, horizontal_lines)
            vector = compare_grid_cells_and_compute_vectors(cells)
            dx = vector[4]
            dy = vector[5]


            pattern[numero_piston] = 70
            relax_miroir()
            set_all_pistons(70)

            time.sleep(1)
            
        print("✅ Matrice d'interaction terminée.")

    # Création et lancement du thread
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
