# analyse_spots.py (ou le nom de votre fichier d'analyse)
import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
import time

# --- IMPORTER LA FONCTION DE CAPTURE DE LA CAMÉRA ---
from Acquisition import get_live_image  # Assurez-vous que acquisition.py est dans le même dossier ou dans le PYTHONPATH

# --- Paramètres ---
GRID_ROWS = 7
GRID_COLS = 7
VECTOR_SCALE = 1
THRESHOLD = 0.5
MIN_AREA = 20


# === Fonctions utilitaires (inchangées) ===
def detect_spots(image, threshold, min_area):
    """Détecte les centres lumineux dans l'image"""
    norm_image = (image - image.min()) / (image.max() - image.min())
    binary_image = (norm_image > threshold).astype(np.uint8)
    labeled_image, num_features = label(binary_image)
    centers = []
    for i in range(1, num_features + 1):
        region = (labeled_image == i)
        if np.sum(region) >= min_area:
            centers.append(center_of_mass(region))
    return np.array(centers)


def recadrage_centers(centers, image_shape, margin_ratio=1.5):
    """Recadre l'image en alignant le spot central avec le centre géométrique de la grille"""
    y_median = np.median(centers[:, 0])
    x_median = np.median(centers[:, 1])
    dist_y = np.median(np.abs(centers[:, 0] - y_median))
    dist_x = np.median(np.abs(centers[:, 1] - x_median))
    margin_y = int(dist_y * margin_ratio * (GRID_ROWS // 2))
    margin_x = int(dist_x * margin_ratio * (GRID_COLS // 2))
    y_center = int(y_median)
    x_center = int(x_median)
    y_min = max(0, y_center - margin_y)
    y_max = min(image_shape[0], y_center + margin_y)
    x_min = max(0, x_center - margin_x)
    x_max = min(image_shape[1], image_shape[1] + margin_x)  # Bugfix here for x_max
    if y_max - y_min < 2 * margin_y:
        if y_min == 0:
            y_max = min(image_shape[0], 2 * margin_y)
        else:
            y_min = max(0, image_shape[0] - 2 * margin_y)
    if x_max - x_min < 2 * margin_x:
        if x_min == 0:
            x_max = min(image_shape[1], 2 * margin_x)
        else:
            x_min = max(0, image_shape[1] - 2 * margin_x)
    return y_min, y_max, x_min, x_max


def generate_grid(mean_center, median_dx, median_dy, grid_rows=7, grid_cols=7):
    """Construction d'une grille régulière de cases"""
    grid_boxes = []
    grid_centers = []
    offset_y = grid_rows // 2
    offset_x = grid_cols // 2
    for i in range(-offset_y, offset_y + 1):
        for j in range(-offset_x, offset_x + 1):
            cy = mean_center[0] + i * median_dy
            cx = mean_center[1] + j * median_dx
            top = int(cy - median_dy // 2)
            left = int(cx - median_dx // 2)
            grid_boxes.append([left, top, left + median_dx, top + median_dy])
            grid_centers.append([cy, cx])
    return np.array(grid_centers), grid_boxes


# === Fonctions principales ===
# La fonction mesure_et_enregistre_grille doit accepter une image NumPy directement
def mesure_et_enregistre_grille(image, threshold, grid_file):
    """Mesure et enregistre la grille avec vérification du centrage à partir d'une image NumPy."""
    if image.ndim == 3:
        image = image[..., 0]

    centers = detect_spots(image, threshold, MIN_AREA)
    if len(centers) < GRID_ROWS * GRID_COLS:
        raise RuntimeError(f"Nombre de spots insuffisant: {len(centers)}/{GRID_ROWS * GRID_COLS}")

    y_min, y_max, x_min, x_max = recadrage_centers(centers, image.shape)
    cropped_image = image[y_min:y_max, x_min:x_max]
    adjusted_centers = centers - [y_min, x_min]

    sorted_centers = adjusted_centers[np.lexsort((adjusted_centers[:, 1], adjusted_centers[:, 0]))]
    grid_center_image_geometric = np.array([cropped_image.shape[0] / 2, cropped_image.shape[1] / 2])
    distances = np.linalg.norm(sorted_centers - grid_center_image_geometric, axis=1)
    closest_indices = np.argsort(distances)[:GRID_ROWS * GRID_COLS]
    selected_centers = sorted_centers[closest_indices]
    selected_centers = selected_centers[np.lexsort((selected_centers[:, 1], selected_centers[:, 0]))]
    try:
        grid = selected_centers.reshape(GRID_ROWS, GRID_COLS, 2)
    except Exception as e:
        raise RuntimeError(f"Impossible d'organiser {len(selected_centers)} centres en grille 7x7: {e}")

    center_spot = grid[GRID_ROWS // 2, GRID_COLS // 2, :]
    middle_row = grid[GRID_ROWS // 2, :, :]
    middle_col = grid[:, GRID_COLS // 2, :]
    dx = np.median(np.diff(middle_row[:, 1]))
    dy = np.median(np.diff(middle_col[:, 0]))

    grid_centers_theoretical = []
    grid_boxes_theoretical = []
    offset_y_grid = GRID_ROWS // 2
    offset_x_grid = GRID_COLS // 2

    for i in range(-offset_y_grid, offset_y_grid + 1):
        for j in range(-offset_x_grid, offset_x_grid + 1):
            cy_theoretical = center_spot[0] + i * dy
            cx_theoretical = center_spot[1] + j * dx
            grid_centers_theoretical.append([cy_theoretical, cx_theoretical])
            grid_boxes_theoretical.append([
                int(cx_theoretical - dx / 2),
                int(cy_theoretical - dy / 2),
                int(cx_theoretical + dx / 2),
                int(cy_theoretical + dy / 2)
            ])

    alignment_offset = center_spot - grid_center_image_geometric
    print("\n=== ANALYSE DE CENTRAGE ===")
    print(f"Spot central détecté : (y={center_spot[0]:.1f}, x={center_spot[1]:.1f})")
    print(f"Centre géométrique image: (y={grid_center_image_geometric[0]:.1f}, x={grid_center_image_geometric[1]:.1f})")
    print(
        f"Décalage             : (Δy={alignment_offset[0]:.1f} px, Δx={alignment[1]:.1f} px)")  # Bug: alignment -> alignment_offset

    if np.linalg.norm(alignment_offset) > 5:
        print("ATTENTION: Décalage important (>5 px) détecté entre le spot central et le centre de l'image!")

    plt.figure(figsize=(12, 10))
    plt.imshow(cropped_image, cmap='gray')
    for box in grid_boxes_theoretical:
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=1, edgecolor='lime', facecolor='none'
        ))
    central_box_theoretical = grid_boxes_theoretical[offset_y_grid * GRID_COLS + offset_x_grid]
    plt.gca().add_patch(plt.Rectangle(
        (central_box_theoretical[0], central_box_theoretical[1]),
        central_box_theoretical[2] - central_box_theoretical[0],
        central_box_theoretical[3] - central_box_theoretical[1],
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    ))
    plt.scatter(center_spot[1], center_spot[0], c='cyan', marker='x', s=200, label='Spot central détecté')
    plt.scatter(grid_center_image_geometric[1], grid_center_image_geometric[0], c='magenta', marker='+', s=200,
                label='Centre géométrique image')
    plt.axhline(grid_center_image_geometric[0], color='white', linestyle=':', alpha=0.5)
    plt.axvline(grid_center_image_geometric[1], color='white', linestyle=':', alpha=0.5)
    bbox = dict(boxstyle="round", fc="white", ec="black", alpha=0.8)
    plt.annotate(f"Spot: ({center_spot[1]:.1f}, {center_spot[0]:.1f})",
                 xy=(center_spot[1], center_spot[0]), xytext=(10, 20),
                 textcoords='offset points', bbox=bbox)
    plt.annotate(f"Centre Im: ({grid_center_image_geometric[1]:.1f}, {grid_center_image_geometric[0]:.1f})",
                 xy=(grid_center_image_geometric[1], grid_center_image_geometric[0]), xytext=(10, -30),
                 textcoords='offset points', bbox=bbox)
    plt.legend()
    plt.title("Vérification du centrage de la grille (Grille théorique centrée sur le spot détecté)")
    plt.show()

    grid_data = {
        "mean_center": center_spot.tolist(),
        "median_dx": float(dx),
        "median_dy": float(dy),
        "grid_centers": np.array(grid_centers_theoretical).tolist(),
        "grid_boxes": grid_boxes_theoretical,
        "crop_bounds": [y_min, y_max, x_min, x_max],
        "original_shape": image.shape,
        "alignment_offset": alignment_offset.tolist(),
        "num_rows": GRID_ROWS,
        "num_cols": GRID_COLS
    }
    with open(grid_file, 'w') as f:
        json.dump(grid_data, f, indent=4)
    print(f"Grille sauvegardée dans {grid_file}")
    return cropped_image, grid_boxes_theoretical


def charge_grille(grid_file):
    """Charge une grille existante depuis un fichier JSON"""
    with open(grid_file, 'r') as f:
        data = json.load(f)
    required_keys = ["mean_center", "median_dx", "median_dy", "grid_centers", "grid_boxes", "crop_bounds"]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"Fichier '{grid_file}' incomplet. Clés manquantes: {missing}")
    data["grid_centers"] = np.array(data["grid_centers"])
    data["mean_center"] = np.array(data["mean_center"])
    return data


def mesure_centres_gravite(image, grid_data, threshold):
    """
    Mesure les centres de gravité par rapport à la grille de référence pour une seule image.
    Retourne les déplacements et le masque de validité.
    """
    if image.ndim == 3:
        image = image[..., 0]

    y_min, y_max, x_min, x_max = grid_data["crop_bounds"]
    cropped_image = image[y_min:y_max, x_min:x_max]
    grid_boxes = grid_data["grid_boxes"]
    grid_centers_theoretical = np.array(grid_data["grid_centers"], dtype=float)

    centers_measured = []
    valid_mask = []

    for box in grid_boxes:
        left, top, right, bottom = box
        sub_image = cropped_image[top:bottom, left:right]
        sub_image = np.where(sub_image > threshold, sub_image, 0)

        if np.sum(sub_image) > 0:
            cy, cx = center_of_mass(sub_image)
            centers_measured.append([top + cy, left + cx])
            valid_mask.append(True)
        else:
            centers_measured.append([np.nan, np.nan])
            valid_mask.append(False)

    centers_measured = np.array(centers_measured, dtype=float)
    valid_mask = np.array(valid_mask)

    displacements = []
    matched_spots_theoretical = []

    for i, (gy, gx) in enumerate(grid_centers_theoretical):
        if valid_mask[i]:
            measured = centers_measured[i]
            displacement = measured - np.array([gy, gx])
            displacements.append(displacement)
            matched_spots_theoretical.append([gy, gx])

    displacements = np.array(displacements)
    matched_spots_theoretical = np.array(matched_spots_theoretical)

    return cropped_image, displacements, matched_spots_theoretical, valid_mask, grid_boxes


def mesure_dynamique_flux_images(camera_function, grid_file, threshold, num_frames=50, update_interval=0.1):
    """
    Mesure dynamique des centres de gravité à partir d'un flux d'images en direct.
    Affiche les déplacements en temps réel et les enregistre.

    Args:
        camera_function (callable): Une fonction qui, lorsqu'appelée, renvoie une nouvelle image (tableau NumPy).
                                    Ex: `get_live_image` de votre module `acquisition`.
        grid_file (str): Chemin du fichier JSON de la grille de référence.
        threshold (float): Seuil pour la détection des spots.
        num_frames (int): Nombre de cadres à traiter (pour limiter la durée de l'exécution).
        update_interval (float): Intervalle de mise à jour de l'affichage en secondes.
    """
    # 1. Chargement de la grille de référence
    if not os.path.exists(grid_file):
        print("Grille de référence non trouvée. Lancement de l'étalonnage initial...")
        # Capture la première image pour l'étalonnage
        initial_image = camera_function()
        if initial_image is None:
            raise RuntimeError("Impossible de capturer l'image initiale pour l'étalonnage.")
        mesure_et_enregistre_grille(initial_image, threshold, grid_file)

    grid_data = charge_grille(grid_file)
    print(f"Grille de référence chargée depuis {grid_file}")

    # Préparation de l'affichage dynamique
    plt.ion()  # Active le mode interactif de matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    img_display = None
    quiver_plot = None

    # Dessine la grille de référence une seule fois
    for box in grid_data["grid_boxes"]:
        left, top, right, bottom = box
        rect = plt.Rectangle((left, top), right - left, bottom - top,
                             linewidth=1, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    ax.set_title("Mesure de Déplacements en Temps Réel")
    ax.set_aspect('equal')
    plt.tight_layout()

    all_displacements = []

    # 2. Boucle de traitement des images en direct
    print("\nLancement de la mesure dynamique des déplacements en direct...")
    for i in range(num_frames):
        image = camera_function()  # Capture une nouvelle image en direct
        if image is None:
            print(f"Avertissement: Impossible de capturer l'image {i + 1}. Skipping.")
            all_displacements.append([])
            time.sleep(update_interval)
            continue  # Passe au cadre suivant

        try:
            cropped_image, displacements, matched_spots_theoretical, valid_mask, _ = \
                mesure_centres_gravite(image, grid_data, threshold)

            # --- Affichage dynamique ---
            if img_display is None:
                img_display = ax.imshow(cropped_image, cmap='gray')
            else:
                img_display.set_data(cropped_image)
                # Assurez-vous que les limites d'axes sont fixes si le recadrage est constant
                ax.set_xlim(0, cropped_image.shape[1])
                ax.set_ylim(cropped_image.shape[0], 0)  # Y inversé pour l'affichage image

            if quiver_plot:
                quiver_plot.remove()

            if len(displacements) > 0:
                quiver_plot = ax.quiver(
                    matched_spots_theoretical[:, 1], matched_spots_theoretical[:, 0],
                    displacements[:, 1], displacements[:, 0],
                    angles='xy', scale_units='xy', scale=1 / VECTOR_SCALE,
                    color='red', width=0.005, zorder=3
                )
            else:
                quiver_plot = None

            ax.set_title(f"Déplacements en Temps Réel - Cadre {i + 1} ({np.sum(valid_mask)} spots détectés)")
            fig.canvas.draw()
            fig.canvas.flush_events()
            print(f"Cadre {i + 1}: Mesures traitées. Spots détectés: {np.sum(valid_mask)}")

            all_displacements.append(displacements.tolist())

        except Exception as e:
            print(f"Erreur lors du traitement du cadre {i + 1}: {e}")
            all_displacements.append([])
            continue

        time.sleep(update_interval)

    plt.ioff()
    plt.show()

    output_displacements_file = "all_displacements_history.json"
    with open(output_displacements_file, 'w') as f:
        json.dump(all_displacements, f, indent=4)
    print(f"\nHistorique de tous les déplacements sauvegardé dans {output_displacements_file}")

    return all_displacements


# === Point d'entrée ===
if __name__ == "__main__":
    grid_reference_file = "grille_reference.json"

    # --- Lancement du processus ---
    mesure_dynamique_flux_images(
        get_live_image,  # Passe la fonction de capture de la caméra
        grid_reference_file,
        THRESHOLD,
        num_frames=100,#apture 100 images en direct
        update_interval=0.1  # Environ 10 images par seconde (1/0.05)
    )

