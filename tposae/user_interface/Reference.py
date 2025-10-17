import cv2
import numpy as np

def detect_spots(image, threshold=0.12, min_area=10):
    """Détecte les centres lumineux dans l'image"""
    # Normalisation de l'image entre 0 et 1
    norm_image = (image - image.min()) / (image.max() - image.min())
    
    # Créer une image binaire en fonction du seuil
    binary_image = (norm_image > threshold).astype(np.uint8)

    # Appliquer un flou pour réduire le bruit
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Trouver les contours
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []  # Liste pour stocker les centres des spots
    for contour in contours:
        # Filtrer les petites zones qui peuvent être du bruit
        if cv2.contourArea(contour) >= min_area:
            # Calculer le centre du contour
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                # Calcul du centre basé sur les moments
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                centers.append(center)
    
    # Retourner un array NumPy des centres
    return np.array(centers)

def save_coordinates_to_file(x_min, x_max, y_min, y_max, filename='reference.txt'):
    """Enregistre les coordonnées xmin, xmax, ymin, ymax dans un fichier texte."""
    with open(filename, 'w') as file:
        file.write(f"x_min: {x_min}\n")
        file.write(f"x_max: {x_max}\n")
        file.write(f"y_min: {y_min}\n")
        file.write(f"y_max: {y_max}\n")
    print(f"✅ Coordonnées sauvegardées dans '{filename}'")

def process_and_save_images(image_path):
    """
    Charge l'image de référence, détecte les spots lumineux et 
    enregistre deux images :
    - L'image originale 'image_reference.jpg'
    - L'image zoomée et marquée 'image_reference_centre.jpg'
    """
    # Charger l'image
    image = cv2.imread(image_path)

    # Vérifier si l'image a été chargée
    if image is None:
        print("❌ Impossible de charger l'image.")
        return

    # Convertir l'image en niveaux de gris pour la détection des spots
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les spots dans l'image
    spots_centers = detect_spots(gray_image)

    # Vérifier si des centres ont été détectés
    if len(spots_centers) == 0:
        print("Aucun centre de spot détecté.")
        return

    # Trouver les coordonnées des spots
    print("Centres détectés : ", np.array(spots_centers))

    # Trouver la taille de la matrice de microlentilles
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
        # Ajuster les coordonnées du centre pour l'image zoomée
        adjusted_center = (center[0] - x_min, center[1] - y_min)
        # Remplacer le pixel correspondant à ce centre par du rouge
        image_zoom[adjusted_center[1], adjusted_center[0]] = [0, 0, 255]  # [B, G, R] : couleur rouge

    # Sauvegarder l'image originale (image_reference.jpg)
    cv2.imwrite('image_reference.jpg', image)

    # Sauvegarder l'image zoomée et marquée (image_reference_centre.jpg)
    cv2.imwrite('image_reference_centre.jpg', image_zoom)

    print(f"✅ Image originale sauvegardée sous 'image_reference.jpg'")
    print(f"✅ Image zoomée avec centres détectés sauvegardée sous 'image_reference_centre.jpg'")

# Chemin de l'image de référence
image_path = 'image_reference.jpg'

# Traiter et sauvegarder les deux images
process_and_save_images(image_path)
