from Reference import detect_spots, read_grid_cells_from_reference, assign_coordinates_from_file
import numpy as np
# Assurez-vous d'importer les fonctions nécessaires de votre fichier Reference.py si vous en avez besoin (par exemple, detect_spots)

import cv2
import numpy as np
from scipy.ndimage import center_of_mass
# Vous aurez besoin de la fonction read_grid_cells_from_reference (déjà fournie)
# et de la fonction get_global_roi_offset (fournie ci-dessous).

# --- Fonction utilitaire pour lire l'Offset du ROI Global ---
def get_global_roi_offset(filename='reference.txt'):
    """
    Lit x_min et y_min pour obtenir le décalage global du ROI.
    (Ces valeurs sont soustraites des coordonnées absolues de la grille pour
     obtenir les indices de l'image rognée).
    """
    x_min_global, y_min_global = 0, 0
    
    # Utilisation de votre fonction assign_coordinates_from_file si elle est disponible
    coords = assign_coordinates_from_file(filename) 
    
    if coords:
        # Assurez-vous que la conversion en int fonctionne pour l'indexation
        x_min_global = int(coords.get('x_min', 0))
        y_min_global = int(coords.get('y_min', 0))
        
    return x_min_global, y_min_global


def detect_spots_cells(image, grid_cells, filename_ref='reference.txt', threshold=0.15):
    """
    Calcule le Centre de Masse (CoM) des spots en utilisant directement les limites 
    des cellules de grille comme masque (méthode de CoM direct).

    Args:
        image (np.array): Image (déjà rognée par le ROI global).
        grid_cells (list): Liste de dictionnaires des cellules de grille (coordonnées absolues).
        filename_ref (str): Nom du fichier de référence pour obtenir l'offset global du ROI.
        threshold (float): Seuil de binarisation (normalisé).

    Returns:
        tuple: (np.array(centers), list(valid_contours))
    """
    # 1. Pré-calculs (Offset et Normalisation)
    # Ceci est OBLIGATOIRE si l'image 'image' est rognée et que 'grid_cells' utilise des coordonnées absolues
    x_offset, y_offset = get_global_roi_offset(filename_ref)
    
    # Normalisation de l'image
    if image.max() == image.min():
         norm_image = np.zeros_like(image, dtype=float)
    else:
        norm_image = (image - image.min()) / (image.max() - image.min())
    
    centers = []
    valid_contours = []
    
    # Binarisation globale de l'image normalisée
    binary_full_image = (norm_image > threshold).astype(np.uint8)

    # 2. Itération sur les cellules de référence
    for cell in grid_cells:
        # On ne traite que les cellules qui contenaient un spot dans la référence
        if cell['has_spot']:
            
            # 3. Calcul des limites de la cellule DANS l'espace de l'image rognée (AVEC DÉCALAGE)
            y_start = int(max(0, cell['y_min'] - y_offset))
            y_end = int(min(image.shape[0], cell['y_max'] - y_offset))
            x_start = int(max(0, cell['x_min'] - x_offset))
            x_end = int(min(image.shape[1], cell['x_max'] - x_offset))
            
            if x_end <= x_start or y_end <= y_start:
                continue
                
            # 4. Extraire le ROI de l'image binaire
            binary_roi = binary_full_image[y_start:y_end, x_start:x_end]
            
            # Vérifier s'il y a de l'intensité (au moins un pixel au-dessus du seuil)
            if np.sum(binary_roi) == 0:
                continue 

            # 5. Calcul du Centre de Masse (CoM) sur la masse binaire DANS le ROI
            # center_of_mass retourne (cy_roi, cx_roi)
            cy_roi, cx_roi = center_of_mass(binary_roi) 
            
            # 6. Conversion en coordonnées de l'image rognée (décalage = x_start et y_start)
            # Les coordonnées renvoyées sont celles que votre code de visualisation attend
            cx_image_roi = int(round(cx_roi)) + x_start
            cy_image_roi = int(round(cy_roi)) + y_start
            
            centers.append((cx_image_roi, cy_image_roi))
            
            # 7. Création d'un contour factice pour la compatibilité
            contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                biggest_contour_roi = max(contours, key=cv2.contourArea)
                
                # Décaler les points du contour pour les ramener aux coordonnées de l'image rognée
                shifted_contour = biggest_contour_roi + (x_start, y_start)
                valid_contours.append(shifted_contour)

    return np.array(centers), valid_contours

def compute_cells_from_grid_ref(centers, vertical_lines, horizontal_lines):
    """
    Calcule les informations des cellules (cx, cy, has_spot, limites, xspot, yspot)
    en utilisant les lignes de grille fournies par le fichier de référence.
    
    Args:
        centers (np.array): Liste des centres de spots réels détectés (x, y).
        vertical_lines (list): Liste des lignes verticales de la grille.
        horizontal_lines (list): Liste des lignes horizontales de la grille.
        
    Returns:
        list: Liste des dictionnaires de cellules.
    """
    if not vertical_lines or not horizontal_lines:
        print("❌ Les lignes de grille sont manquantes.")
        return []

    # 1. Extraction des coordonnées des lignes de grille
    # Vertical_lines est (x1, y1, x2, y2), où x1=x2 est la coordonnée x de la ligne.
    # On prend toutes les coordonnées x distinctes.
    vertical_x_coords = sorted(list(set([v[0] for v in vertical_lines])))
    
    # Horizontal_lines est (x1, y1, x2, y2), où y1=y2 est la coordonnée y de la ligne.
    # On prend toutes les coordonnées y distinctes.
    horizontal_y_coords = sorted(list(set([h[1] for h in horizontal_lines])))

    # Vérification minimale: Il faut au moins 2 lignes verticales et 2 lignes horizontales pour former une cellule.
    if len(vertical_x_coords) < 2 or len(horizontal_y_coords) < 2:
        print("❌ Coordonnées de grille insuffisantes pour définir les cellules.")
        return []

    cells = []
    # Conversion des centres de spots réels en tuples pour l'itération
    centers_tuple = [tuple(c) for c in centers]

    # 2. Itération sur les cellules définies par l'intersection des lignes
    for j in range(len(horizontal_y_coords) - 1):
        for i in range(len(vertical_x_coords) - 1):
            
            # Définition des limites de la cellule (déjà calculées dans reference.txt)
            x_start = vertical_x_coords[i]
            x_end = vertical_x_coords[i+1]
            y_start = horizontal_y_coords[j]
            y_end = horizontal_y_coords[j+1]
            
            # Calcul du centre théorique de la cellule
            center_x = (x_start + x_end) / 2
            center_y = (y_start + y_end) / 2
            
            has_spot = False
            xspot = 0.0 # Utilisation de 0.0 comme valeur par défaut, cohérent avec write_reference_file
            yspot = 0.0 

            # 3. Logique d'appariement par surface (center du spot dans les limites de la case)
            for cx_spot, cy_spot in centers_tuple:
                
                # [x_min, x_max[ et [y_min, y_max[ pour éviter le double comptage sur les bords
                is_in_x = (x_start <= cx_spot < x_end)
                is_in_y = (y_start <= cy_spot < y_end)
                
                if is_in_x and is_in_y:
                    has_spot = True
                    xspot = cx_spot 
                    yspot = cy_spot 
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

    return cells

def compare_grid_cells_and_compute_vectors(current_cells, reference_filename='reference.txt'):
    """
    Compare les cellules de grille actuelles (calculées à l'instant T) avec
    les cellules de grille de référence (lues du fichier) pour calculer le vecteur 
    de déplacement (dx, dy) pour chaque spot.

    Le vecteur va du spot de référence (origine) au spot actuel (fin).

    Args:
        current_cells (list): Liste des dictionnaires de cellules calculées 
                              pour l'image actuelle (via compute_cells_from_grid_ref).
        reference_filename (str): Nom du fichier de référence.
        
    Returns:
        list: Liste des dictionnaires de vecteurs trouvés.
              Format: {
                  'ref_cx': cx de la cellule, 
                  'ref_cy': cy de la cellule, 
                  'ref_xspot': xspot de référence, 
                  'ref_yspot': yspot de référence, 
                  'dx': déplacement en x (actuel - référence), 
                  'dy': déplacement en y (actuel - référence)
              }
    """
    # 1. Lecture des cellules de référence
    ref_cells = read_grid_cells_from_reference(reference_filename)
    
    if not ref_cells:
        print("❌ Impossible de lire les cellules de référence. Arrêt.")
        return []

    # 2. Création d'un dictionnaire de référence pour un accès rapide par (cx, cy)
    ref_dict = {}
    for cell in ref_cells:
        # Clé: centre théorique de la cellule (cx, cy)
        key = (cell['cx'], cell['cy']) 
        ref_dict[key] = cell

    vectors = []
    
    # 3. Comparaison des cellules actuelles avec la référence
    for current_cell in current_cells:
        key = (current_cell['cx'], current_cell['cy'])
        
        # Vérifie si cette cellule existe dans la référence
        if key in ref_dict:
            ref_cell = ref_dict[key]
            
            # 4. Vérification de la présence des spots dans les deux cellules
            # Si un spot était présent dans la référence ET est présent dans l'actuel
            if ref_cell['has_spot'] and current_cell['has_spot']:
                
                # Coordonnées de référence (origine du vecteur)
                ref_xspot = ref_cell['xspot']
                ref_yspot = ref_cell['yspot']
                
                # Coordonnées actuelles (fin du vecteur)
                current_xspot = current_cell['xspot']
                current_yspot = current_cell['yspot']
                
                # 5. Calcul du vecteur de déplacement
                # Le vecteur part de la référence et va vers l'actuel
                # dx = x_fin - x_origine
                dx = current_xspot - ref_xspot
                dy = current_yspot - ref_yspot
                
                # 6. Stockage des informations
                vectors.append({
                    'ref_cx': ref_cell['cx'], 
                    'ref_cy': ref_cell['cy'], 
                    'ref_xspot': ref_xspot, 
                    'ref_yspot': ref_yspot, 
                    'dx': dx, 
                    'dy': dy
                })
        # else: La cellule actuelle correspond à une position théorique sans spot de référence, ignorée pour le calcul de déplacement.
        
    return vectors