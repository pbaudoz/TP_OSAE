import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Détection des spots ---
def detect_spots(image, threshold=0.12, min_area=10):
    norm_image = (image - image.min()) / (image.max() - image.min())
    binary_image = (norm_image > threshold).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return np.array(centers)

# --- Mettre à jour la section centres des spots dans reference.txt ---
def update_reference_centers(reference_file, centers):
    try:
        with open(reference_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    new_lines = []
    inside_section = False

    for line in lines:
        stripped = line.strip()
        if stripped == "#BEGIN centres des spots":
            inside_section = True
            continue
        if stripped == "#END centres des spots":
            inside_section = False
            continue
        if not inside_section:
            new_lines.append(line)

    # Ajouter la section complète au fichier
    if new_lines and new_lines[-1].strip() != "":
        new_lines.append("\n")
    new_lines.append("#BEGIN centres des spots\n")
    for c in centers:
        new_lines.append(f"{c[0]},{c[1]}\n")
    new_lines.append("#END centres des spots\n")

    # Écrire dans le fichier
    with open(reference_file, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ Centres des spots mis à jour dans '{reference_file}'")

# --- Paramètres ---
image_path = "/Users/dorian_pontes_sousa/Documents/AO4OSAE_git/TP_OSAE/image_0.png"
reference_file = "/Users/dorian_pontes_sousa/Documents/AO4OSAE_git/TP_OSAE/reference.txt"
threshold = 0.12
min_area = 10

# --- Exécution ---
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
centers = detect_spots(image, threshold, min_area)

if len(centers) == 0:
    raise RuntimeError("Aucun spot détecté !")

# Mettre à jour le fichier reference.txt
update_reference_centers(reference_file, centers)
