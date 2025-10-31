import cv2
import numpy as np

def detect_spots(image, threshold=0.12, min_area=10):
    """Détecte les centres lumineux dans l'image"""
    norm_image = (image - image.min()) / (image.max() - image.min())
    binary_image = (norm_image > threshold).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center = (int(moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"]))
                centers.append(center)
    return np.array(centers)

def save_coordinates_to_file(x_min, x_max, y_min, y_max, filename='reference.txt'):
    """Enregistre les coordonnées xmin, xmax, ymin, ymax dans un fichier texte."""
    with open(filename, 'w') as file:
        file.write(f"x_min: {x_min}\n")
        file.write(f"x_max: {x_max}\n")
        file.write(f"y_min: {y_min}\n")
        file.write(f"y_max: {y_max}\n")
    print(f"✅ Coordonnées sauvegardées dans '{filename}'")

def save_grid_to_file(spots_centers, cell_size=50, filename='reference.txt'):
    """Ajoute la grille (cellule + centres) au fichier de référence"""
    try:
        with open(filename, 'a') as file:  # 'a' pour ajouter
            file.write(f"grid_cell_size: {cell_size}\n")
            centers_str = ",".join([f"({x},{y})" for x, y in spots_centers])
            file.write(f"grid_centers: {centers_str}\n")
        print(f"✅ Grille sauvegardée dans '{filename}'")
    except Exception as e:
        print(f"❌ Impossible de sauvegarder la grille : {e}")

def draw_grid_around_spots(image, spots_centers, cell_size=50):
    """Dessine une grille autour des spots sur l'image"""
    grid_image = image.copy()
    for center in spots_centers:
        cx, cy = center
        top_left = (max(0, cx - cell_size//2), max(0, cy - cell_size//2))
        bottom_right = (min(image.shape[1]-1, cx + cell_size//2),
                        min(image.shape[0]-1, cy + cell_size//2))
        cv2.rectangle(grid_image, top_left, bottom_right, (0,255,0), 1)
        cv2.circle(grid_image, (cx, cy), 2, (0,0,255), -1)
    return grid_image

def process_and_save_images(image_path, cell_size=50):
    """Détecte les spots et sauvegarde les images + référence + grille"""
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Impossible de charger l'image.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    spots_centers = detect_spots(gray_image)

    if len(spots_centers) == 0:
        print("Aucun centre de spot détecté.")
        return

    # Coordonnées zone utile
    x_min = max(0, np.min(spots_centers[:,0]) - 50)
    x_max = min(image.shape[1], np.max(spots_centers[:,0]) + 50)
    y_min = max(0, np.min(spots_centers[:,1]) - 50)
    y_max = min(image.shape[0], np.max(spots_centers[:,1]) + 50)
    save_coordinates_to_file(x_min, x_max, y_min, y_max)

    # Enregistrer grille
    save_grid_to_file(spots_centers, cell_size)

    # Image zoomée avec centres marqués
    image_zoom = image[y_min:y_max, x_min:x_max].copy()
    for center in spots_centers:
        adjusted_center = (center[0]-x_min, center[1]-y_min)
        image_zoom[adjusted_center[1], adjusted_center[0]] = [0,0,255]

    cv2.imwrite('image_reference.jpg', image)
    cv2.imwrite('image_reference_centre.jpg', image_zoom)
    print("✅ Images sauvegardées : 'image_reference.jpg' et 'image_reference_centre.jpg'")
