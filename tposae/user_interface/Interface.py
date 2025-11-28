import sys
import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QCheckBox, QVBoxLayout, 
    QHBoxLayout, QLineEdit, QFrame, QGroupBox, QSizePolicy, 
    QSlider, QSpinBox # Ajout de QSlider et QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import QTimer, Qt

from visualisation import get_live_image, set_camera_roi, set_camera_exposure, capture_and_save_image
from Reference import process_and_save_images, assign_coordinates_from_file, read_grid_from_reference


class LiveReferenceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TP OSAE - Interface de Visualisation et Référence")
        self.setGeometry(100, 100, 1000, 650) # Taille initiale de la fenêtre
        
        self.setStyleSheet(self._get_qstyle())

        # --- Variables d'état et initialisation ---
        self.coords = assign_coordinates_from_file('reference.txt')
        if self.coords:
            # Initialiser le ROI de la caméra
            set_camera_roi(self.coords['x_min'], self.coords['y_min'],
                           self.coords['x_max'] - self.coords['x_min'],
                           self.coords['y_max'] - self.coords['y_min'])
            print(f"✅ ROI défini : {self.coords}")
        else:
            self.coords = {'x_min': 0, 'y_min': 0, 'x_max': 500, 'y_max': 500} 

        self.vertical_lines, self.horizontal_lines = read_grid_from_reference('reference.txt')
        self.show_grid = False
        self.grid_pixmap = None
        self.LIVE_WIDTH = 500
        self.LIVE_HEIGHT = 500
        
        # Exposition de base en microsecondes (0.0002 s = 200 µs)
        self.BASE_EXPOSURE_US = 200 # <-- Changé de 20000 à 200
        self.MAX_EXPOSURE_US = 1000000 # 1 seconde
        self.MIN_EXPOSURE_US = 10 # 10 microsecondes

        # --- Configuration de l'UI ---
        self._setup_ui()

        # --- Initialisation des éléments graphiques ---
        self.create_grid_pixmap(self.LIVE_WIDTH, self.LIVE_HEIGHT)
        self._init_exposure_slider() # Initialisation du slider

        # --- Démarrage des Timers ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_image)
        self.timer.start(30) # ~33 fps pour la vidéo live

        self.ref_timer = QTimer()
        self.ref_timer.timeout.connect(self.update_reference_image)
        self.ref_timer.start(1000) # Mise à jour de la référence chaque seconde
        
        self.show()

    def _get_qstyle(self):
        """Définit le style CSS pour les widgets PyQt5."""
        return """
            QWidget {
                background-color: #2e2e2e;
                color: #e0e0e0;
                font-family: Arial;
                font-size: 10pt;
            }
            QGroupBox {
                border: 2px solid #5a5a5a;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
                font-size: 11pt;
                font-weight: bold;
                color: #4CAF50; /* Vert pour les titres de groupe */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
            QLabel#live_display {
                border: 1px solid #5a5a5a;
                background-color: #000000;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #4CAF50; /* Vert */
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 10pt;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit, QSpinBox {
                border: 1px solid #5a5a5a;
                padding: 5px;
                border-radius: 3px;
                background-color: #3e3e3e;
            }
            QCheckBox {
                spacing: 5px;
            }
            /* Style pour le QSlider */
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px; 
                background: #4e4e4e;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """

    def _setup_ui(self):
        """Configure la disposition des widgets dans la fenêtre."""
        main_layout = QHBoxLayout(self)
        
        # --- Colonne 1 : Affichage Live et Commandes (à Gauche) ---
        live_commands_col = QVBoxLayout()
        
        # 1. Groupe Vidéo Live
        live_group = QGroupBox("Vidéo Live et Grille")
        live_group_layout = QVBoxLayout(live_group)
        
        self.live_label = QLabel()
        self.live_label.setObjectName("live_display") # Utiliser pour le style CSS
        self.live_label.setFixedSize(self.LIVE_WIDTH, self.LIVE_HEIGHT) 
        self.live_label.setAlignment(Qt.AlignCenter)
        live_group_layout.addWidget(self.live_label)

        self.show_grid_checkbox = QCheckBox("Afficher la grille de référence")
        self.show_grid_checkbox.stateChanged.connect(self.toggle_grid)
        live_group_layout.addWidget(self.show_grid_checkbox)

        live_commands_col.addWidget(live_group)
        
        # 2. Groupe Commandes
        commands_group = QGroupBox("Commandes Caméra et Référence")
        commands_layout = QVBoxLayout(commands_group)
        
        # Bouton Acquisition Référence
        self.acquire_button = QPushButton("Acquérir Nouvelle Référence") # Retrait du "1."
        self.acquire_button.clicked.connect(self.acquire_new_reference_image)
        commands_layout.addWidget(self.acquire_button)

        # Contrôle d'Exposition (Slider + SpinBox)
        exposure_group = QGroupBox("Exposition")
        exposure_group_layout = QVBoxLayout(exposure_group)
        
        # Affichage de la valeur en microsecondes (µs)
        self.exposure_display = QSpinBox()
        self.exposure_display.setSuffix(" µs")
        self.exposure_display.setRange(self.MIN_EXPOSURE_US, self.MAX_EXPOSURE_US)
        self.exposure_display.setSingleStep(100) # Pas de 100 µs
        
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(self.MIN_EXPOSURE_US, self.MAX_EXPOSURE_US)
        
        # Connexion mutuelle : slider -> spinbox, spinbox -> slider
        self.exposure_slider.valueChanged.connect(self.exposure_display.setValue)
        self.exposure_display.valueChanged.connect(self.exposure_slider.setValue)
        # La mise à jour de l'exposition de la caméra se fait dès que la valeur change (via spinbox)
        self.exposure_display.valueChanged.connect(self.update_exposure) 

        exposure_group_layout.addWidget(self.exposure_display)
        exposure_group_layout.addWidget(self.exposure_slider)
        commands_layout.addWidget(exposure_group)
        
        live_commands_col.addWidget(commands_group)
        live_commands_col.addStretch(1) # Pousser les éléments vers le haut

        main_layout.addLayout(live_commands_col)

        # --- Colonne 2 : Image de Référence (à Droite) ---
        
        ref_group = QGroupBox("Image de Référence Détectée")
        ref_layout = QVBoxLayout(ref_group)
        
        self.ref_label = QLabel("Pas d'image de référence")
        self.ref_label.setFixedSize(250, 250)
        self.ref_label.setAlignment(Qt.AlignCenter)
        self.ref_label.setScaledContents(True) # Pour le KeepAspectRatio de la référence
        ref_layout.addWidget(self.ref_label)
        
        ref_group.setFixedWidth(300) # Fixer la largeur de la colonne de référence

        main_layout.addWidget(ref_group)

    def _init_exposure_slider(self):
        """Initialise le slider et la valeur d'exposition."""
        # Fixer la valeur initiale (0.0002 s = 200 µs)
        self.exposure_slider.setValue(self.BASE_EXPOSURE_US)
        # Ceci va automatiquement déclencher self.update_exposure() via la connexion

    # --- Logique de la Grille ---
    def toggle_grid(self):
        self.show_grid = self.show_grid_checkbox.isChecked()
        self.update_live_image()

    def create_grid_pixmap(self, width, height):
        """Crée le QPixmap de la grille avec la translation et le scaling (Anti-Crash inclus)."""
        
        coords = assign_coordinates_from_file('reference.txt')
        if not coords:
            self.grid_pixmap = None
            return
            
        x_min, y_min = coords['x_min'], coords['y_min']
        roi_width = coords['x_max'] - x_min
        roi_height = coords['y_max'] - y_min
        
        # VÉRIFICATION ANTI-CRASH
        if roi_width <= 0 or roi_height <= 0:
            print("❌ AVERTISSEMENT GRILLE : Largeur/hauteur ROI nulle ou invalide. La grille ne sera pas affichée.")
            self.grid_pixmap = None
            return

        scale_x = width / roi_width
        scale_y = height / roi_height

        self.grid_pixmap = QPixmap(width, height)
        self.grid_pixmap.fill(QColor(0, 0, 0, 0))

        painter = QPainter(self.grid_pixmap)
        pen = QPen(QColor(0, 255, 0, 200)) # Vert
        pen.setWidth(1)
        painter.setPen(pen)

        def transform_and_draw_line(line):
            """Applique la transformation (Translation + Scaling) et dessine la ligne."""
            
            x1_scaled = (line[0] - x_min) * scale_x
            y1_scaled = (line[1] - y_min) * scale_y
            x2_scaled = (line[2] - x_min) * scale_x
            y2_scaled = (line[3] - y_min) * scale_y
            
            if math.isnan(x1_scaled) or math.isinf(x1_scaled) or math.isnan(y1_scaled) or math.isinf(y1_scaled):
                 return 

            painter.drawLine(int(x1_scaled), int(y1_scaled), int(x2_scaled), int(y2_scaled))

        for v in self.vertical_lines:
            transform_and_draw_line(v)

        for h in self.horizontal_lines:
            transform_and_draw_line(h)

        painter.end()


    # --- Mise à jour de la vidéo live ---
    def update_live_image(self):
        frame = get_live_image()
        if frame is not None:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Utilisation de Qt.IgnoreAspectRatio pour que l'image Remplisse EXACTEMENT le 500x500
            pixmap = pixmap.scaled(self.live_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            # Superposition de la grille
            if self.show_grid and self.grid_pixmap and not self.grid_pixmap.isNull():
                painter = QPainter(pixmap)
                painter.drawPixmap(0, 0, self.grid_pixmap)
                painter.end()

            self.live_label.setPixmap(pixmap)

    # --- Mise à jour de l'image de référence ---
    def update_reference_image(self):
        ref_image = cv2.imread('image_reference_centre.jpg')
        if ref_image is not None:
            if len(ref_image.shape) == 2:
                ref_image = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
            h, w, ch = ref_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(ref_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
            
            pixmap = QPixmap.fromImage(qt_image)
            # KeepAspectRatio est préférable pour la référence
            pixmap = pixmap.scaled(self.ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.ref_label.setPixmap(pixmap)

    # --- Acquisition d'une nouvelle image de référence ---
    def acquire_new_reference_image(self):
        print("🔵 Acquisition d’une nouvelle image de référence...")
        self.acquire_button.setEnabled(False) # Désactiver pour éviter les doubles clics
        QApplication.processEvents() # Forcer la mise à jour de l'UI
        
        try:
            capture_and_save_image()
            process_and_save_images('image_reference.jpg')
            
            self.coords = assign_coordinates_from_file('reference.txt')
            if self.coords:
                set_camera_roi(self.coords['x_min'], self.coords['y_min'],
                               self.coords['x_max'] - self.coords['x_min'],
                               self.coords['y_max'] - self.coords['y_min'])
                print(f"🎯 Nouveau ROI appliqué : {self.coords}")
                
            self.vertical_lines, self.horizontal_lines = read_grid_from_reference('reference.txt')
            self.create_grid_pixmap(self.live_label.width(), self.live_label.height())
            self.update_reference_image()

        finally:
            self.acquire_button.setEnabled(True) # Réactiver le bouton

    # --- Mise à jour de l'exposition ---
    def update_exposure(self, value_us):
        """Met à jour l'exposition de la caméra en microsecondes (µs).
        La valeur 'value_us' est fournie directement par le QSpinBox."""
        try:
            # La valeur 'value_us' vient directement du QSpinBox (en microsecondes)
            set_camera_exposure(value_us)
            # Affichage en secondes pour le terminal
            print(f"✅ Exposition réglée sur {value_us / 1e6:.6f} s ({value_us} µs)")
        except Exception as e:
            print(f"❌ Erreur exposition : {e}")
            
    # --- Appliquer zoom ROI ---
    def apply_zoom_roi(self):
        # Cette fonction est conservée pour la compatibilité avec l'ancienne logique
        # mais n'est plus liée à un bouton dans l'UI. 
        self.coords = assign_coordinates_from_file('reference.txt')
        if self.coords:
            set_camera_roi(self.coords['x_min'], self.coords['y_min'],
                           self.coords['x_max'] - self.coords['x_min'],
                           self.coords['y_max'] - self.coords['y_min'])
            print(f"🎯 ROI de Zoom appliqué : {self.coords}")
            
            self.vertical_lines, self.horizontal_lines = read_grid_from_reference('reference.txt')
            self.create_grid_pixmap(self.live_label.width(), self.live_label.height())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveReferenceWindow()
    sys.exit(app.exec_())