import cv2
import numpy as np

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

def save_coordinates_to_file(x_min, x_max, y_min, y_max, filename='reference.txt'):
    """Enregistre les coordonnées xmin, xmax, ymin, ymax dans un fichier texte."""
    with open(filename, 'w') as file:
        file.write(f"x_min: {x_min}\n")
        file.write(f"x_max: {x_max}\n")
        file.write(f"y_min: {y_min}\n")
        file.write(f"y_max: {y_max}\n")
    print(f"✅ Coordonnées sauvegardées dans '{filename}'")

def assign_coordinates_from_file(filename='reference.txt'):
    """Lit les coordonnées à partir d'un fichier texte."""
    coords = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if ":" in line:
                    key, val = line.split(":")
                    coords[key.strip()] = int(val.strip())
        return coords
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des coordonnées : {e}")
        return None

def process_and_save_images(image_path):
    """Charge l'image de référence, détecte les spots lumineux et enregistre les images."""
    # Charger l'image
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Impossible de charger l'image.")
        return

    # Convertir l'image en niveaux de gris pour la détection des spots
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les spots dans l'image
    spots_centers = detect_spots(gray_image)

    if len(spots_centers) == 0:
        print("Aucun centre de spot détecté.")
        return

    # Trouver les coordonnées des spots
    x_min = np.min(spots_centers[:, 0]) - 50
    x_max = np.max(spots_centers[:, 0]) + 50
    y_min = np.min(spots_centers[:, 1]) - 50
    y_max = np.max(spots_centers[:, 1]) + 50

    # Assurer que les indices sont dans les limites de l'image
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)

    # Sauvegarder les coordonnées dans un fichier texte
    save_coordinates_to_file(x_min, x_max, y_min, y_max)

    # Nouveau découpage de l'image
    image_zoom = image[y_min:y_max, x_min:x_max]

    # Afficher l'image zoomée avec les centres détectés
    for center in spots_centers:
        adjusted_center = (center[0] - x_min, center[1] - y_min)
        image_zoom[adjusted_center[1], adjusted_center[0]] = [0, 0, 255]  # Marque le centre en rouge

    # Sauvegarder les images
    cv2.imwrite('image_reference.jpg', image)
    cv2.imwrite('image_reference_centre.jpg', image_zoom)

    print(f"✅ Image originale sauvegardée sous 'image_reference.jpg'")
    print(f"✅ Image zoomée sauvegardée sous 'image_reference_centre.jpg'")

