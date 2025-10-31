import cv2
import numpy as np

def detect_spots(image, threshold=0.12, min_area=10):
    norm_image = (image - image.min()) / (image.max() - image.min())
    binary_image = (norm_image > threshold).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                centers.append(center)
    return np.array(centers)

def save_coordinates_to_file(x_min, x_max, y_min, y_max, grid_lines, filename='reference.txt'):
    with open(filename, 'w') as file:
        file.write(f"x_min: {x_min}\n")
        file.write(f"x_max: {x_max}\n")
        file.write(f"y_min: {y_min}\n")
        file.write(f"y_max: {y_max}\n")
        for i, (x1, y1, x2, y2) in enumerate(grid_lines):
            file.write(f"grid_line_{i}: {x1},{y1},{x2},{y2}\n")
    print(f"✅ Coordonnées et grille sauvegardées dans '{filename}'")

def generate_grid(x_min, x_max, y_min, y_max, nx, ny):
    """Crée une grille homogène avec nx x ny cases."""
    grid_lines = []
    dx = (x_max - x_min) // nx
    dy = (y_max - y_min) // ny
    # Lignes verticales
    for i in range(nx + 1):
        x = x_min + i * dx
        grid_lines.append((x, y_min, x, y_max))
    # Lignes horizontales
    for j in range(ny + 1):
        y = y_min + j * dy
        grid_lines.append((x_min, y, x_max, y))
    return grid_lines

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

    x_min = np.min(spots_centers[:, 0]) - 50
    x_max = np.max(spots_centers[:, 0]) + 50
    y_min = np.min(spots_centers[:, 1]) - 50
    y_max = np.max(spots_centers[:, 1]) + 50

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)

    # Détermination du nombre de cases : racine carrée du nombre de spots détectés
    n_spots = len(spots_centers)
    n_grid = int(np.sqrt(n_spots))
    grid_lines = generate_grid(x_min, x_max, y_min, y_max, n_grid, n_grid)

    save_coordinates_to_file(x_min, x_max, y_min, y_max, grid_lines)

    image_zoom = image[y_min:y_max, x_min:x_max]
    for (x1, y1, x2, y2) in grid_lines:
        cv2.line(image_zoom, (x1 - x_min, y1 - y_min), (x2 - x_min, y2 - y_min), (0, 255, 0), 1)

    cv2.imwrite('image_reference.jpg', image)
    cv2.imwrite('image_reference_centre.jpg', image_zoom)
    print("✅ Image et grille enregistrées.")
