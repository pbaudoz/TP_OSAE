import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass

# --- Détection des spots ---
def detect_spots(image, threshold=0.12, min_area=10):
    """Détecte les centres lumineux dans l'image en utilisant l'étiquetage et le Centre de Masse (CoM)"""
    
    # 1. Normalisation et seuillage binaire
    # Votre méthode de normalisation actuelle est conservée
    norm_image = (image - image.min()) / (image.max() - image.min())
    binary_image = (norm_image > threshold).astype(np.uint8)

    # 2. Étiquetage des régions connectées
    # 'label' identifie chaque groupe de pixels connectés (spots)
    # L'image étiquetée est un tableau où chaque spot a une valeur entière unique (> 0)
    labeled_image, num_features = label(binary_image)
    
    centers = []

    # 3. Calcul du Centre de Masse pour chaque région (spot)
    for i in range(1, num_features + 1):
        # Créer un masque booléen pour la région courante
        region_mask = (labeled_image == i)
        
        # Vérifier si la surface du spot est suffisante
        if np.sum(region_mask) >= min_area:
            # center_of_mass retourne (y, x) dans l'indexation NumPy
            cy, cx = center_of_mass(region_mask)
            
            # On stocke sous forme (x, y) pour la cohérence avec OpenCV
            centers.append((int(cx), int(cy)))
            
    return np.array(centers)

# --- Fonctions inchangées ---
# ... (write_reference_file, assign_coordinates_from_file, compute_grid_from_centers, process_and_save_images, read_grid_from_reference)
# Le reste de votre code (write_reference_file, assign_coordinates_from_file, compute_grid_from_centers, process_and_save_images, read_grid_from_reference)
# n'a pas besoin d'être modifié car il utilise le résultat de 'detect_spots' (une liste de centres (x, y)).

# --- Écriture du fichier de référence ---
def write_reference_file(reference_file, x_min, x_max, y_min, y_max,
                         centers, vertical_lines, horizontal_lines):
    lines = []
    lines.append(f"x_min: {x_min}\n")
    lines.append(f"x_max: {x_max}\n")
    lines.append(f"y_min: {y_min}\n")
    lines.append(f"y_max: {y_max}\n\n")

    # Centres des spots
    lines.append("#BEGIN centres des spots\n")
    for c in centers:
        lines.append(f"{c[0]},{c[1]}\n")
    lines.append("#END centres des spots\n\n")

    # Grille
    lines.append("#BEGIN grille\n")
    lines.append("# Lignes verticales : x1,y1,x2,y2\n")
    for v in vertical_lines:
        lines.append(f"{v[0]},{v[1]},{v[2]},{v[3]}\n")
    lines.append("# Lignes horizontales : x1,y1,x2,y2\n")
    for h in horizontal_lines:
        lines.append(f"{h[0]},{h[1]},{h[2]},{h[3]}\n")
    lines.append("#END grille\n")

    with open(reference_file, 'w') as f:
        f.writelines(lines)
    print("📄 reference.txt mis à jour proprement")

# --- Lecture ROI ---
def assign_coordinates_from_file(filename='reference.txt'):
    coords = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, val = line.split(":")
                if val.strip().isdigit():
                    coords[key.strip()] = int(val.strip())
        return coords
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des coordonnées : {e}")
        return None

# --- Calcul grille ---
def compute_grid_from_centers(centers):
    if len(centers) < 4:
        return [], []
    y_median = np.median(centers[:,1])
    x_median = np.median(centers[:,0])
    line_central = centers[np.abs(centers[:,1]-y_median) < 5]
    line_central = line_central[np.argsort(line_central[:,0])]
    col_central = centers[np.abs(centers[:,0]-x_median) < 5]
    col_central = col_central[np.argsort(col_central[:,1])]
    dx = np.median(np.diff(line_central[:,0]))
    dy = np.median(np.diff(col_central[:,1]))
    y_top, y_bottom = col_central[0,1]-dy/2, col_central[-1,1]+dy/2
    x_left, x_right = line_central[0,0]-dx/2, line_central[-1,0]+dx/2
    vertical_lines = [
        ((line_central[i,0]+line_central[i+1,0])/2, y_top,
         (line_central[i,0]+line_central[i+1,0])/2, y_bottom)
        for i in range(len(line_central)-1)
    ]
    vertical_lines.insert(0, (x_left, y_top, x_left, y_bottom))
    vertical_lines.append((x_right, y_top, x_right, y_bottom))
    horizontal_lines = [
        (x_left, (col_central[i,1]+col_central[i+1,1])/2,
         x_right, (col_central[i,1]+col_central[i+1,1])/2)
        for i in range(len(col_central)-1)
    ]
    horizontal_lines.insert(0, (x_left, y_top, x_right, y_top))
    horizontal_lines.append((x_left, y_bottom, x_right, y_bottom))
    return vertical_lines, horizontal_lines

# --- Traitement et sauvegarde image ---
def process_and_save_images(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Impossible de charger l'image.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    spots_centers = detect_spots(gray_image)
    if len(spots_centers) == 0:
        print("Aucun centre de spot détecté.")
        return

    # Calcul ROI
    x_min = max(0, np.min(spots_centers[:, 0]) - 50)
    x_max = min(image.shape[1], np.max(spots_centers[:, 0]) + 50)
    y_min = max(0, np.min(spots_centers[:, 1]) - 50)
    y_max = min(image.shape[0], np.max(spots_centers[:, 1]) + 50)

    # Calcul grille (pour référence)
    vertical_lines, horizontal_lines = compute_grid_from_centers(spots_centers)
    write_reference_file("reference.txt", x_min, x_max, y_min, y_max,
                         spots_centers, vertical_lines, horizontal_lines)

    # Création image zoomée avec centres en rouge
    image_zoom = image[y_min:y_max, x_min:x_max].copy()
    for center in spots_centers:
        adjusted_center = (center[0] - x_min, center[1] - y_min)
        cv2.circle(image_zoom, adjusted_center, radius=3, color=(0, 0, 255), thickness=-1)

    # Sauvegarde
    cv2.imwrite('image_reference.jpg', image)
    cv2.imwrite('image_reference_centre.jpg', image_zoom)

    print(f"✅ Image originale sauvegardée sous 'image_reference.jpg'")
    print(f"✅ Image zoomée avec centres sauvegardée sous 'image_reference_centre.jpg'")

def read_grid_from_reference(filename='reference.txt'):
    vertical_lines = []
    horizontal_lines = []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        in_grid = False
        reading_v = False
        reading_h = False

        for line in lines:
            line = line.strip()

            # Activer/désactiver sections
            if line == "#BEGIN grille":
                in_grid = True
                continue
            if line == "#END grille":
                break

            # Sauter tout tant qu'on n'est pas dans la section grille
            if not in_grid:
                continue

            # Gestion des sous-sections
            if line.startswith("# Lignes verticales"):
                reading_v = True
                reading_h = False
                continue
            if line.startswith("# Lignes horizontales"):
                reading_h = True
                reading_v = False
                continue

            # Ignorer commentaires ou lignes vides
            if line.startswith("#") or line == "":
                continue

            # Convertir uniquement les lignes "x1,y1,x2,y2"
            try:
                parts = [float(x) for x in line.split(",")]
                if len(parts) != 4:
                    continue

                if reading_v:
                    vertical_lines.append(tuple(parts))
                elif reading_h:
                    horizontal_lines.append(tuple(parts))

            except ValueError:
                # Ligne non convertible → on ignore
                continue

        return vertical_lines, horizontal_lines

    except Exception as e:
        print(f"❌ Erreur lecture grille : {e}")
        return [], []