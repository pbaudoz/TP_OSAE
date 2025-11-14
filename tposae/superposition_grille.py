# overlay_grid_from_reference.py
import cv2
import matplotlib.pyplot as plt

def read_grid_from_reference(reference_file):
    vertical_lines = []
    horizontal_lines = []
    inside_section = False
    current_section = None

    with open(reference_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped == "#BEGIN grille":
                inside_section = True
                continue
            if stripped == "#END grille":
                inside_section = False
                break
            if inside_section:
                if stripped.startswith("# Lignes verticales"):
                    current_section = 'v'
                    continue
                if stripped.startswith("# Lignes horizontales"):
                    current_section = 'h'
                    continue
                if stripped and current_section:
                    x1, y1, x2, y2 = map(float, stripped.split(","))
                    if current_section == 'v':
                        vertical_lines.append((x1, y1, x2, y2))
                    else:
                        horizontal_lines.append((x1, y1, x2, y2))
    return vertical_lines, horizontal_lines

def overlay_grid(image_path, reference_file, x_min, x_max, y_min, y_max):
    vertical_lines, horizontal_lines = read_grid_from_reference(reference_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='gray')

    # Tracer la grille
    for line in vertical_lines:
        plt.plot([line[0], line[2]], [line[1], line[3]], 'lime', linewidth=1)
    for line in horizontal_lines:
        plt.plot([line[0], line[2]], [line[1], line[3]], 'lime', linewidth=1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_max, y_min)
    plt.title("Grille superposée depuis reference.txt")
    plt.show()

# --- Paramètres ---
image_path = "/Users/dorian_pontes_sousa/Documents/AO4OSAE_git/TP_OSAE/image_0.png"
reference_file = "/Users/dorian_pontes_sousa/Documents/AO4OSAE_git/TP_OSAE/reference.txt"
x_min, x_max = 279, 530
y_min, y_max = 97, 347

# --- Exécution ---
overlay_grid(image_path, reference_file, x_min, x_max, y_min, y_max)
