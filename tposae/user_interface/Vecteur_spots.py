from Reference import detect_spots, read_grid_cells_from_reference
import numpy as np
# Assurez-vous d'importer les fonctions nécessaires de votre fichier Reference.py si vous en avez besoin (par exemple, detect_spots)

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