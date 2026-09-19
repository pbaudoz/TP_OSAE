import cv2
import numpy as np
import math
from scipy.ndimage import label, center_of_mass 

# --- Détection des spots (inchangée) ---
def detect_spots(image, threshold=0.12, min_area=10):
    """
    Détecte les centres lumineux dans l'image en utilisant la labellisation et le centre de masse
    et crée des contours factices pour la compatibilité avec la visualisation.
    """
    norm_image = (image - image.min()) / (image.max() - image.min())
    binary_image = (norm_image > threshold).astype(np.uint8)
    
    # Étape 1: Labellisation des régions (Scipy)
    labeled_image, num_features = label(binary_image)
    
    centers = []
    valid_contours = [] 
    
    # Itération sur chaque région labellisée
    for i in range(1, num_features + 1):
        region = (labeled_image == i)
        
        if np.sum(region) >= min_area:
            # Étape 2: Calcul du Centre de Masse (y, x)
            cy, cx = center_of_mass(region) 
            
            # center_of_mass retourne (y, x), nous le convertissons en (cx, cy) ou (x, y)
            cx = int(round(cx)) 
            cy = int(round(cy))
            
            centers.append((cx, cy))
            
            # Étape 3: Création d'un contour factice pour la compatibilité
            region_uint8 = region.astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                biggest_contour = max(contours, key=cv2.contourArea)
                valid_contours.append(biggest_contour)

    return np.array(centers), valid_contours

# --- Calcul grille et cellules (MODIFIÉE pour inclure xspot, yspot) ---
def compute_grid_and_cells(centers):
    """Calcule les lignes de la grille et détermine l'état (spot présent/absent) de chaque cellule 
    en vérifiant si le centre du spot est dans les limites rectangulaires de la cellule et stocke ses coordonnées."""
    if len(centers) < 4:
        return [], [], [] 

    y_median = np.median(centers[:,1])
    x_median = np.median(centers[:,0])
    line_central = centers[np.abs(centers[:,1]-y_median) < 5]
    line_central = line_central[np.argsort(line_central[:,0])]
    col_central = centers[np.abs(centers[:,0]-x_median) < 5]
    col_central = col_central[np.argsort(col_central[:,1])]

    if len(line_central) < 2 or len(col_central) < 2:
        return [], [], []

    dx = np.median(np.diff(line_central[:,0]))
    dy = np.median(np.diff(col_central[:,1]))

    if dx == 0 or dy == 0:
        return [], [], []

    y_top, y_bottom = col_central[0,1]-dy/2, col_central[-1,1]+dy/2
    x_left, x_right = line_central[0,0]-dx/2, line_central[-1,0]+dx/2

    vertical_x_coords = [(line_central[i,0] + line_central[i+1,0]) / 2 for i in range(len(line_central)-1)]
    vertical_x_coords.insert(0, x_left)
    vertical_x_coords.append(x_right)

    vertical_lines = [
        (x_coord, y_top, x_coord, y_bottom)
        for x_coord in vertical_x_coords
    ]
    
    horizontal_y_coords = [(col_central[i,1] + col_central[i+1,1]) / 2 for i in range(len(col_central)-1)]
    horizontal_y_coords.insert(0, y_top)
    horizontal_y_coords.append(y_bottom)

    horizontal_lines = [
        (x_left, y_coord, x_right, y_coord)
        for y_coord in horizontal_y_coords
    ]
    
    cells = []
    centers_tuple = [tuple(c) for c in centers]
    
    for j in range(len(horizontal_y_coords) - 1):
        for i in range(len(vertical_x_coords) - 1):
            x_start = vertical_x_coords[i]
            x_end = vertical_x_coords[i+1]
            y_start = horizontal_y_coords[j]
            y_end = horizontal_y_coords[j+1]
            
            center_x = (x_start + x_end) / 2
            center_y = (y_start + y_end) / 2
            
            has_spot = False
            xspot = None # Initialisation des coordonnées du spot réel
            yspot = None 

            # Logique d'appariement par surface
            for cx_spot, cy_spot in centers_tuple:
                
                is_in_x = (x_start + 2 <= cx_spot < x_end - 2)
                is_in_y = (y_start + 2 <= cy_spot < y_end - 2)
                
                if is_in_x and is_in_y:
                    has_spot = True
                    xspot = cx_spot # Stockage des coordonnées du spot
                    yspot = cy_spot # Stockage des coordonnées du spot
                    break 
                    
            cells.append({
                'cx': center_x, 
                'cy': center_y, 
                'has_spot': has_spot,
                'x_min': x_start,
                'y_min': y_start,
                'x_max': x_end,
                'y_max': y_end,
                'xspot': xspot, 
                'yspot': yspot 
            })

    return vertical_lines, horizontal_lines, cells

# --- Écriture du fichier de référence (MODIFIÉE pour xspot/yspot) ---
def write_reference_file(reference_file, x_min, x_max, y_min, y_max,
                            centers, vertical_lines, horizontal_lines, cells):
    """Écrit les coordonnées du ROI, des centres et de la grille dans le fichier de référence."""
    lines = []
    lines.append(f"x_min: {x_min}\n")
    lines.append(f"x_max: {x_max}\n")
    lines.append(f"y_min: {y_min}\n")
    lines.append(f"y_max: {y_max}\n\n")

    lines.append("#BEGIN centres des spots\n")
    for c in centers:
        lines.append(f"{c[0]},{c[1]}\n")
    lines.append("#END centres des spots\n\n")

    lines.append("#BEGIN grille\n")
    lines.append("# Lignes verticales : x1,y1,x2,y2\n")
    for v in vertical_lines:
        lines.append(f"{v[0]},{v[1]},{v[2]},{v[3]}\n")
    lines.append("# Lignes horizontales : x1,y1,x2,y2\n")
    for h in horizontal_lines:
        lines.append(f"{h[0]},{h[1]},{h[2]},{h[3]}\n")
    lines.append("#END grille\n\n")
    
    lines.append("#BEGIN cellules de grille\n")
    # Nouveau format
    lines.append("# Format : cx,cy,has_spot(0/1),x_min,y_min,x_max,y_max,xspot,yspot\n") 
    
    for c in cells:
        # Convertir None en 0.0 pour l'écriture dans le fichier si le spot est absent
        xspot_val = c['xspot'] if c['xspot'] is not None else 0.0 
        yspot_val = c['yspot'] if c['yspot'] is not None else 0.0
        
        lines.append(f"{c['cx']},{c['cy']},{int(c['has_spot'])},{c['x_min']},{c['y_min']},{c['x_max']},{c['y_max']},{xspot_val},{yspot_val}\n")
    lines.append("#END cellules de grille\n")

    with open(reference_file, 'w') as f:
        f.writelines(lines)
    print("📄 reference.txt mis à jour proprement")

# --- Lecture des fonctions (MODIFIÉE pour xspot/yspot) ---
def read_grid_cells_from_reference(filename='reference.txt'):
    """Lit les données des cellules de grille depuis le fichier de référence."""
    cells = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        in_cells_section = False
        
        for line in lines:
            line = line.strip()

            if line == "#BEGIN cellules de grille":
                in_cells_section = True
                continue
            if line == "#END cellules de grille":
                break

            if not in_cells_section or line.startswith("#") or line == "":
                continue

            try:
                parts = [float(x) for x in line.split(",")]
                # Le format attend maintenant 9 parties
                if len(parts) == 9: 
                    cells.append({
                        'cx': parts[0], 
                        'cy': parts[1], 
                        'has_spot': bool(int(parts[2])),
                        'x_min': parts[3],
                        'y_min': parts[4],
                        'x_max': parts[5],
                        'y_max': parts[6],
                        'xspot': parts[7], 
                        'yspot': parts[8]  
                    })
            except ValueError:
                continue

        return cells

    except Exception as e:
        print(f"❌ Erreur lecture cellules : {e}") 
        return []

def assign_coordinates_from_file(filename='reference.txt'):
    """Lit les coordonnées x_min, x_max, y_min, y_max du fichier de référence."""
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

def read_grid_from_reference(filename='reference.txt'):
    """Lit les lignes de la grille (verticales et horizontales) depuis le fichier de référence."""
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

            if line == "#BEGIN grille":
                in_grid = True
                continue
            if line == "#END grille":
                break

            if not in_grid:
                continue

            if line.startswith("# Lignes verticales"):
                reading_v = True
                reading_h = False
                continue
            if line.startswith("# Lignes horizontales"):
                reading_h = True
                reading_v = False
                continue

            if line.startswith("#") or line == "":
                continue

            try:
                parts = [float(x) for x in line.split(",")]
                if len(parts) != 4:
                    continue

                if reading_v:
                    vertical_lines.append(tuple(parts))
                elif reading_h:
                    horizontal_lines.append(tuple(parts))

            except ValueError:
                continue

        return vertical_lines, horizontal_lines

    except Exception as e:
        print(f"❌ Erreur lecture grille : {e}")
        return [], []
        
# --- Traitement et sauvegarde image (inchangée) ---
def process_and_save_images(image_path):
    print(f"🔬 Début du traitement de l'image pour la référence : {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Impossible de charger l'image à {image_path}. Vérifiez le chemin et l'extension.")
        return

    # Gestion des images à 1 canal (gris) ou 3 canaux (BGR)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray_image = image
    else:
        print("❌ Format d'image non pris en charge.")
        return
        
    spots_centers, spots_contours = detect_spots(gray_image) 
    
    if len(spots_centers) == 0:
        print("Aucun centre de spot détecté. 🛑 Arrêt du traitement.")
        return

    # Calcul du ROI de zoom autour des spots
    spots_centers_np = np.array(spots_centers)
    x_min = max(0, np.min(spots_centers_np[:, 0]) - 25)
    x_max = min(image.shape[1], np.max(spots_centers_np[:, 0]) + 25)
    y_min = max(0, np.min(spots_centers_np[:, 1]) - 25)
    y_max = min(image.shape[0], np.max(spots_centers_np[:, 1]) + 25)
    
    vertical_lines, horizontal_lines, cells = compute_grid_and_cells(spots_centers_np) 
    
    write_reference_file("reference.txt", x_min, x_max, y_min, y_max,
                            spots_centers, vertical_lines, horizontal_lines, cells)

    image_zoom = image[y_min:y_max, x_min:x_max].copy()
    
    # Dimensions de l'image zoomée
    H_zoom, W_zoom = image_zoom.shape[:2]
    # Couleur BGR (Bleu) pour OpenCV
    BLUE_PIXEL = [255, 0, 0] 
    
    for i, center in enumerate(spots_centers):
        contour = spots_contours[i]
        
        # Ajustement des coordonnées au ROI zoomé
        contour_adj = contour - [x_min, y_min]
        x_center_adj = center[0] - x_min
        y_center_adj = center[1] - y_min
        
        # A. Dessiner le contour non rempli en rouge (épaisseur 1)
        cv2.drawContours(image_zoom, [contour_adj], -1, (0, 0, 255), 1) 

        # B. DESSINER LE CENTRE DE MASSE EN TANT QUE 1 PIXEL BLEU
        # On utilise l'indexation directe du tableau NumPy pour garantir un point de 1 pixel
        if 0 <= y_center_adj < H_zoom and 0 <= x_center_adj < W_zoom:
            image_zoom[y_center_adj, x_center_adj] = BLUE_PIXEL
            
    # Sauvegarde des images en PNG
    cv2.imwrite('image_reference.png', image)
    
    # SAUVEGARDE DE L'IMAGE CENTRÉE
    cv2.imwrite('image_reference_centre.png', image_zoom)

    print(f"✅ Image originale sauvegardée sous 'image_reference.png'")
    print(f"✅ Image zoomée avec contours rouges (fins) et centre bleu (1 pixel) sauvegardée sous 'image_reference_centre.png' (Format sans perte)")