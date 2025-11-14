# compute_grid_from_reference_with_image.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_centers_from_reference(reference_file):
    centers = []
    inside_section = False
    with open(reference_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped == "#BEGIN centres des spots":
                inside_section = True
                continue
            if stripped == "#END centres des spots":
                inside_section = False
                continue
            if inside_section and stripped:
                x, y = map(float, stripped.split(","))
                centers.append((x, y))
    if not centers:
        raise RuntimeError("Aucun centre de spot trouvé dans la section '#BEGIN centres des spots'.")
    return np.array(centers)

def compute_grid_from_centers(centers):
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

    vertical_lines = [((line_central[i,0]+line_central[i+1,0])/2, y_top,
                       (line_central[i,0]+line_central[i+1,0])/2, y_bottom)
                      for i in range(len(line_central)-1)]
    vertical_lines.insert(0, (x_left, y_top, x_left, y_bottom))
    vertical_lines.append((x_right, y_top, x_right, y_bottom))

    horizontal_lines = [ (x_left, (col_central[i,1]+col_central[i+1,1])/2,
                          x_right, (col_central[i,1]+col_central[i+1,1])/2)
                         for i in range(len(col_central)-1)]
    horizontal_lines.insert(0, (x_left, y_top, x_right, y_top))
    horizontal_lines.append((x_left, y_bottom, x_right, y_bottom))

    return vertical_lines, horizontal_lines

def update_reference_grid(reference_file, vertical_lines, horizontal_lines):
    try:
        with open(reference_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    new_lines = []
    inside_section = False

    for line in lines:
        stripped = line.strip()
        if stripped == "#BEGIN grille":
            inside_section = True
            continue
        if stripped == "#END grille":
            inside_section = False
            continue
        if not inside_section:
            new_lines.append(line)

    # Ajouter la nouvelle section complète
    if new_lines and new_lines[-1].strip() != "":
        new_lines.append("\n")
    new_lines.append("#BEGIN grille\n")
    new_lines.append("# Lignes verticales : x1,y1,x2,y2\n")
    for v in vertical_lines:
        new_lines.append(f"{v[0]},{v[1]},{v[2]},{v[3]}\n")
    new_lines.append("# Lignes horizontales : x1,y1,x2,y2\n")
    for h in horizontal_lines:
        new_lines.append(f"{h[0]},{h[1]},{h[2]},{h[3]}\n")
    new_lines.append("#END grille\n")

    with open(reference_file, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ Section 'grille' mise à jour dans '{reference_file}'")

def tracer_grille_on_image(centers, vertical_lines, horizontal_lines, image_path, x_min, x_max, y_min, y_max):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='gray')
    plt.scatter(centers[:,0], centers[:,1], c='red', s=5, label='Centres spots')

    for line in vertical_lines:
        plt.plot([line[0], line[2]], [line[1], line[3]], 'lime', linewidth=1)
    for line in horizontal_lines:
        plt.plot([line[0], line[2]], [line[1], line[3]], 'lime', linewidth=1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_max, y_min)
    plt.title("Grille calculée depuis les centres des spots")
    plt.legend()
    plt.show()

# --- Paramètres ---
reference_file = "/Users/dorian_pontes_sousa/Documents/AO4OSAE_git/TP_OSAE/reference.txt"
image_path = "/Users/dorian_pontes_sousa/Documents/AO4OSAE_git/TP_OSAE/image_0.png"
x_min, x_max = 279, 530
y_min, y_max = 97, 347

# --- Exécution ---
centers = read_centers_from_reference(reference_file)
vertical_lines, horizontal_lines = compute_grid_from_centers(centers)
update_reference_grid(reference_file, vertical_lines, horizontal_lines)
tracer_grille_on_image(centers, vertical_lines, horizontal_lines, image_path, x_min, x_max, y_min, y_max)
