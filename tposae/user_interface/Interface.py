import sys
import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QCheckBox, QVBoxLayout, 
    QHBoxLayout, QLineEdit, QFrame, QGroupBox, QSizePolicy, 
    QSlider, QSpinBox, QTextEdit, QDoubleSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSignal 

# NOTE: Assurez-vous que ces fonctions existent dans vos modules locaux
from visualisation import get_live_image, set_camera_roi, set_camera_exposure, capture_and_save_image
from Reference import detect_spots, process_and_save_images, assign_coordinates_from_file, read_grid_from_reference, read_grid_cells_from_reference 
from Vecteur_spots import compute_cells_from_grid_ref, compare_grid_cells_and_compute_vectors, detect_spots_cells
# --- MISE À JOUR IMPORT MIROIR : Ajout de set_all_pistons et matrice_interaction ---
from miroir import initialisation_miroir, relax_miroir, set_all_pistons, matrice_interaction 

# --- NOUVELLE CLASSE POUR LA JOURNALISATION ---
class LogHandler(QObject):
    new_text = pyqtSignal(str)
    def write(self, text):
        # Évite d'émettre des chaînes vides si on reçoit juste des espaces/newlines non significatifs
        if text.strip() or text == '\n':
            self.new_text.emit(text)
    def flush(self):
        pass
# --- FIN CLASSE LOGHANDLER ---


class LiveReferenceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TP OSAE - Interface de Visualisation et Référence")
        self.setGeometry(100, 100, 1200, 800) 
        self.setStyleSheet(self._get_qstyle())

        # --- Variables ---
        self.LIVE_WIDTH = 500
        self.LIVE_HEIGHT = 500
        self.BASE_EXPOSURE_US = 200 
        self.MAX_EXPOSURE_US = 1000000 
        self.MIN_EXPOSURE_US = 10
        self.is_paused = False
        self.show_vectors = False 
        self.current_vectors = [] 
        self.vector_scale_factor = 1.0

        self.coords = assign_coordinates_from_file('reference.txt')
        self.status_log = None 
        self.log_handler = None 

        self._setup_ui() 
        self._setup_logging() 

        # --- Initialisation du miroir dès le lancement ---
        try:
            initialisation_miroir()
            print("✅ Miroir initialisé au lancement.")
        except Exception as e:
            print(f"❌ Erreur initialisation miroir : {e}")

        if self.coords:
            set_camera_roi(self.coords['x_min'], self.coords['y_min'],
                           self.coords['x_max'] - self.coords['x_min'],
                           self.coords['y_max'] - self.coords['y_min'])
            print(f"✅ ROI défini : {self.coords}")
        else:
            self.coords = {'x_min': 0, 'y_min': 0, 'x_max': 500, 'y_max': 500} 
            print("❌ Pas de coordonnées de référence trouvées. Utilisation du ROI par défaut.")

        self.vertical_lines, self.horizontal_lines = read_grid_from_reference('reference.txt')
        self.grid_cells = read_grid_cells_from_reference('reference.txt')
        self.show_grid = False
        self.grid_pixmap = None

        self.create_grid_pixmap(self.LIVE_WIDTH, self.LIVE_HEIGHT)
        self._init_exposure_slider() 
        self._init_vector_scale_slider()

        # --- Timers ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_image)
        self.timer.start(30)

        self.ref_timer = QTimer()
        self.ref_timer.timeout.connect(self.update_reference_image)
        self.ref_timer.start(1000)
        
        self.show()

    # ---------------------------
    # UI & Styles
    # ---------------------------
    def _get_qstyle(self):
        return """
            QWidget { background-color: #2e2e2e; color: #e0e0e0; font-family: Arial; font-size: 10pt; }
            QGroupBox { border: 2px solid #5a5a5a; border-radius: 8px; margin-top: 15px; padding-top: 10px; font-size: 11pt; font-weight: bold; color: #4CAF50; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }
            QLabel#live_display { border: 1px solid #5a5a5a; background-color: #000000; border-radius: 5px; }
            QPushButton { background-color: #4CAF50; border: none; color: white; padding: 8px 16px; text-align: center; font-size: 10pt; margin: 4px 2px; border-radius: 4px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton#pause_button { background-color: #FF9800; } 
            QPushButton#pause_button:hover { background-color: #F57C00; } 
            QPushButton#play_button { background-color: #2196F3; }
            QPushButton#play_button:hover { background-color: #1976D2; }
            QLineEdit, QSpinBox, QDoubleSpinBox { border: 1px solid #5a5a5a; padding: 5px; border-radius: 3px; background-color: #3e3e3e; color: #e0e0e0;}
            QCheckBox { spacing: 5px; }
            QSlider::groove:horizontal { border: 1px solid #999999; height: 8px; background: #4e4e4e; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #4CAF50; border: 1px solid #5c5c5c; width: 18px; margin: -5px 0; border-radius: 9px; }
            QTextEdit { background-color: #1e1e1e; color: #b0b0b0; border: 1px solid #5a5a5a; border-radius: 3px; padding: 5px; }
        """

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Colonne Gauche: Vidéo Live et Commandes (Horizontalement séparées) ---
        live_commands_col = QVBoxLayout()
        live_commands_col.setSpacing(15) 

        # 1. Vidéo Live
        live_group = QGroupBox("Vidéo Live et Grille")
        live_group_layout = QVBoxLayout(live_group)
        self.live_label = QLabel()
        self.live_label.setObjectName("live_display")
        self.live_label.setFixedSize(self.LIVE_WIDTH, self.LIVE_HEIGHT)
        self.live_label.setAlignment(Qt.AlignCenter)
        live_group_layout.addWidget(self.live_label)

        viz_hbox = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.setObjectName("pause_button")
        self.pause_button.clicked.connect(self.toggle_live_video)
        viz_hbox.addWidget(self.pause_button)

        self.show_grid_checkbox = QCheckBox("Afficher la grille de référence")
        self.show_grid_checkbox.stateChanged.connect(self.toggle_grid)
        viz_hbox.addWidget(self.show_grid_checkbox)

        self.show_vectors_checkbox = QCheckBox("Afficher les vecteurs (Live)")
        self.show_vectors_checkbox.stateChanged.connect(self.toggle_vectors)
        viz_hbox.addWidget(self.show_vectors_checkbox)

        live_group_layout.addLayout(viz_hbox)
        live_commands_col.addWidget(live_group)
        
        # --- NOUVEAU: Conteneur Horizontal pour les Commandes sous la Vidéo ---
        bottom_commands_hbox = QHBoxLayout()
        
        # --- 2. Bloc de Gauche (Commandes Caméra et Référence) ---
        commands_group = QGroupBox("Commandes Caméra et Référence")
        commands_layout = QVBoxLayout(commands_group)

        self.acquire_button = QPushButton("Acquérir Nouvelle Référence")
        self.acquire_button.clicked.connect(self.acquire_new_reference_image)
        commands_layout.addWidget(self.acquire_button)

        # Exposition
        exposure_group = QGroupBox("Exposition")
        exposure_group_layout = QVBoxLayout(exposure_group)
        self.exposure_display = QSpinBox()
        self.exposure_display.setSuffix(" µs")
        self.exposure_display.setRange(self.MIN_EXPOSURE_US, self.MAX_EXPOSURE_US)
        self.exposure_display.setSingleStep(100)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(self.MIN_EXPOSURE_US, self.MAX_EXPOSURE_US)
        self.exposure_slider.valueChanged.connect(self.exposure_display.setValue)
        self.exposure_display.valueChanged.connect(self.exposure_slider.setValue)
        self.exposure_display.valueChanged.connect(self.update_exposure)
        exposure_group_layout.addWidget(self.exposure_display)
        exposure_group_layout.addWidget(self.exposure_slider)
        commands_layout.addWidget(exposure_group)

        # Échelle des vecteurs
        vector_scale_group = QGroupBox("Échelle des Vecteurs")
        vector_scale_layout = QVBoxLayout(vector_scale_group)
        self.vector_scale_display = QDoubleSpinBox()
        self.vector_scale_display.setPrefix("x ")
        self.vector_scale_display.setRange(1.0, 10.0)
        self.vector_scale_display.setSingleStep(0.5)
        self.vector_scale_display.setValue(self.vector_scale_factor)
        self.vector_scale_slider = QSlider(Qt.Horizontal)
        self.vector_scale_slider.setRange(10, 100) # De 1.0 à 10.0
        self.vector_scale_slider.setValue(int(self.vector_scale_factor * 10))
        self.vector_scale_slider.valueChanged.connect(self._update_vector_scale_from_slider)
        self.vector_scale_display.valueChanged.connect(self._update_vector_scale_from_spinbox)
        vector_scale_layout.addWidget(self.vector_scale_display)
        vector_scale_layout.addWidget(self.vector_scale_slider)
        commands_layout.addWidget(vector_scale_group)

        # Ajout du groupe Caméra/Référence à la partie gauche de la sous-section
        bottom_commands_hbox.addWidget(commands_group)
        
        # --- 3. Bloc de Droite (Commandes Miroir) ---
        mirror_group = QGroupBox("Miroir")
        mirror_layout = QVBoxLayout(mirror_group)
        
        # Bouton Référence du Miroir
        self.set_mirror_ref_button = QPushButton("Référence du Miroir")
        self.set_mirror_ref_button.clicked.connect(self.set_mirror_reference_action)
        mirror_layout.addWidget(self.set_mirror_ref_button)
        
        # Bouton Calcul Matrice d'Interaction
        self.calculate_matrix_button = QPushButton("Calcul Matrice d'Interaction")
        self.calculate_matrix_button.clicked.connect(self.calculate_interaction_matrix_action)
        mirror_layout.addWidget(self.calculate_matrix_button)
        
        # Bouton Relax Miroir
        self.relax_mirror_button = QPushButton("Relaxer Miroir")
        self.relax_mirror_button.clicked.connect(self.relax_miroir_action)
        mirror_layout.addWidget(self.relax_mirror_button)
        
        mirror_layout.addStretch(1) # Pour que les boutons ne prennent pas toute la hauteur
        
        # Ajout du groupe Miroir à la partie droite de la sous-section
        bottom_commands_hbox.addWidget(mirror_group)
        
        # Ajout de la sous-section horizontale à la colonne principale de gauche
        live_commands_col.addLayout(bottom_commands_hbox)

        live_commands_col.addStretch(1) # Pousse les éléments vers le haut
        main_layout.addLayout(live_commands_col)

        # --- Colonne Droite: Image Référence + Log (inchangée) ---
        ref_log_col = QVBoxLayout()
        ref_log_col.setSpacing(15)

        # 1. Image Référence
        ref_group = QGroupBox("Image de Référence Détectée")
        ref_layout = QVBoxLayout(ref_group)
        self.ref_label = QLabel("Pas d'image de référence")
        self.REF_DISPLAY_WIDTH = 600
        self.REF_DISPLAY_HEIGHT = 600
        self.ref_label.setFixedSize(self.REF_DISPLAY_WIDTH, self.REF_DISPLAY_HEIGHT)
        self.ref_label.setAlignment(Qt.AlignCenter)
        self.ref_label.setScaledContents(True)
        ref_layout.addWidget(self.ref_label)
        ref_group.setFixedWidth(self.REF_DISPLAY_WIDTH + 50) 
        ref_log_col.addWidget(ref_group)

        # 2. Log
        log_group = QGroupBox("Status/Journalisation")
        log_layout = QVBoxLayout(log_group)
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setFont(QFont("Monospace", 8))
        self.status_log.setMaximumHeight(200)
        log_layout.addWidget(self.status_log)
        ref_log_col.addWidget(log_group)
        
        ref_log_col.addStretch(1)
        main_layout.addLayout(ref_log_col)
        
    # ---------------------------
    # Logging
    # ---------------------------
    def _setup_logging(self):
        self.log_handler = LogHandler()
        self.log_handler.new_text.connect(self._append_to_log)
        sys.stdout = self.log_handler
    def _append_to_log(self, text):
        self.status_log.insertPlainText(text)
        if not text.endswith('\n'):
            self.status_log.insertPlainText('\n')
        self.status_log.ensureCursorVisible()

    # ---------------------------
    # Sliders
    # ---------------------------
    def _init_exposure_slider(self):
        self.exposure_slider.setValue(self.BASE_EXPOSURE_US)
    def _init_vector_scale_slider(self):
        self.vector_scale_slider.setValue(int(self.vector_scale_factor * 10))
    def _update_vector_scale_from_slider(self, value):
        self.vector_scale_factor = value / 10.0
        self.vector_scale_display.setValue(self.vector_scale_factor)
        self.update_live_image()
        print(f"Échelle des vecteurs réglée sur {self.vector_scale_factor}x")
    def _update_vector_scale_from_spinbox(self, value):
        self.vector_scale_factor = value
        self.vector_scale_slider.setValue(int(value * 10))
        self.update_live_image()

    # ---------------------------
    # Grid & Vectors
    # ---------------------------
    def toggle_grid(self):
        self.show_grid = self.show_grid_checkbox.isChecked()
        self.update_live_image()
    def toggle_vectors(self):
        self.show_vectors = self.show_vectors_checkbox.isChecked()
        self.update_live_image()

    # ---------------------------
    # Actions Miroir
    # ---------------------------
    def relax_miroir_action(self):
        try:
            relax_miroir()
            print("🟢 Miroir relaxé avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de la relaxation du miroir : {e}")
            
    def set_mirror_reference_action(self):
        try:
            set_all_pistons()
            print("⚙️ Référence du miroir (set_all_pistons(0)) appliquée.")
        except Exception as e:
            print(f"❌ Erreur lors de l'application de la référence du miroir : {e}")
            
    # NOUVELLE FONCTION
    def calculate_interaction_matrix_action(self):
        print("⚙️ Démarrage du calcul de la matrice d'interaction...")
        try:
            # Note : Assurez-vous que cette fonction ne bloque pas l'UI trop longtemps
            matrice_interaction() 
            print("✅ Matrice d'interaction calculée et sauvegardée.")
        except Exception as e:
            print(f"❌ Erreur lors du calcul de la matrice d'interaction : {e}")

    def create_grid_pixmap(self, width, height):
        """
        Crée le QPixmap de la grille avec la translation et le scaling. 
        """
        
        coords = assign_coordinates_from_file('reference.txt')
        if not coords:
            self.grid_pixmap = None
            return
            
        x_min, y_min = coords['x_min'], coords['y_min']
        roi_width = coords['x_max'] - x_min
        roi_height = coords['y_max'] - y_min
        
        if roi_width <= 0 or roi_height <= 0:
            print("❌ AVERTISSEMENT GRILLE : Largeur/hauteur ROI nulle ou invalide. La grille ne sera pas affichée.")
            self.grid_pixmap = None
            return

        # Calcul des facteurs d'échelle
        self.scale_x = width / roi_width
        self.scale_y = height / roi_height

        self.grid_pixmap = QPixmap(width, height)
        self.grid_pixmap.fill(QColor(0, 0, 0, 0))

        painter = QPainter(self.grid_pixmap)
        
        # 1. Dessin des lignes de grille (Vert)
        pen_grid = QPen(QColor(0, 255, 0, 200)) # Vert
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)


        def transform_and_draw_line(line):
            """Applique la transformation (Translation + Scaling) et dessine la ligne."""
            
            x1_scaled = (line[0] - x_min) * self.scale_x 
            y1_scaled = (line[1] - y_min) * self.scale_y 
            x2_scaled = (line[2] - x_min) * self.scale_x
            y2_scaled = (line[3] - y_min) * self.scale_y
            
            if math.isnan(x1_scaled) or math.isinf(x1_scaled) or math.isnan(y1_scaled) or math.isinf(y1_scaled):
                return 

            painter.drawLine(int(x1_scaled), int(y1_scaled), int(x2_scaled), int(y2_scaled))

        for v in self.vertical_lines:
            transform_and_draw_line(v)

        for h in self.horizontal_lines:
            transform_and_draw_line(h)

        # 2. Dessin des croix (Rouge) dans les cellules VIDES 
        pen_cross = QPen(QColor(255, 0, 0, 200)) # Rouge
        pen_cross.setWidth(2)
        painter.setPen(pen_cross)
        
        for cell in self.grid_cells:
            if not cell['has_spot']:
                
                x_min_scaled = (cell['x_min'] - x_min) * self.scale_x
                y_min_scaled = (cell['y_min'] - y_min) * self.scale_y
                x_max_scaled = (cell['x_max'] - x_min) * self.scale_x
                y_max_scaled = (cell['y_max'] - y_min) * self.scale_y

                painter.drawLine(int(x_min_scaled), int(y_min_scaled), int(x_max_scaled), int(y_max_scaled))
                painter.drawLine(int(x_min_scaled), int(y_max_scaled), int(x_max_scaled), int(y_min_scaled))

        painter.end()
        
    def draw_vectors(self, painter): # <-- Logique MODIFIÉE avec scale_factor
        """
        Dessine les vecteurs de déplacement (ref_xspot -> ref_xspot + dx*factor) sur le QPainter.
        """
        if not self.show_vectors or not self.current_vectors or not hasattr(self, 'scale_x'):
            return

        x_min, y_min = self.coords.get('x_min', 0), self.coords.get('y_min', 0)
        scale_factor = self.vector_scale_factor # Utilisation du facteur
        
        # Configure le stylo pour les vecteurs
        pen_vector = QPen(QColor(255, 0, 0)) # ROUGE (Rouge Vif)
        pen_vector.setWidth(2)
        
        # Configure le stylo pour les points d'origine (Point de Réf.)
        pen_start_point = QPen(QColor(0, 0, 255)) # Bleu (Blue)
        pen_start_point.setWidth(4)
        
        # Longueur de la pointe de la flèche (en pixels sur l'affichage)
        ARROW_SIZE = 7

        for vector in self.current_vectors:
            
            # 1. Coordonnées de DÉPART (Référence - Full Frame)
            x0_full = vector['ref_xspot']
            y0_full = vector['ref_yspot']
            
            # 2. Coordonnées de FIN (Position actuelle SCALÉE - Full Frame)
            # On applique le facteur au déplacement (dx, dy)
            x1_full_scaled_vec = x0_full + vector['dx'] * scale_factor
            y1_full_scaled_vec = y0_full + vector['dy'] * scale_factor
            
            # --- Conversion en coordonnées LIVE (ROI mis à l'échelle) ---
            
            # Coordonnées de DÉPART (Point Bleu)
            x0_live_scaled = (x0_full - x_min) * self.scale_x
            y0_live_scaled = (y0_full - y_min) * self.scale_y
            
            # Coordonnées de FIN (Pointe Rouge SCALÉE)
            x1_live_scaled_vec = (x1_full_scaled_vec - x_min) * self.scale_x
            y1_live_scaled_vec = (y1_full_scaled_vec - y_min) * self.scale_y
            
            # --- Dessin du Vecteur ---
            
            # Dessin d'un cercle (point) au départ pour indiquer le point de référence
            painter.setPen(pen_start_point)
            painter.drawPoint(int(x0_live_scaled), int(y0_live_scaled))
            
            # Dessin de la ligne principale
            painter.setPen(pen_vector)
            painter.drawLine(int(x0_live_scaled), int(y0_live_scaled), 
                              int(x1_live_scaled_vec), int(y1_live_scaled_vec))
            
            # --- Dessin de la Tête de Flèche (pointant vers la position actuelle SCALÉE) ---
            dx_scaled = x1_live_scaled_vec - x0_live_scaled
            dy_scaled = y1_live_scaled_vec - y0_live_scaled
            length = math.sqrt(dx_scaled**2 + dy_scaled**2)
                
            if length > 0.5: 
                
                # Angle du vecteur par rapport à l'axe X (en radians)
                angle = math.atan2(dy_scaled, dx_scaled)
                
                # Angles pour les deux côtés de la flèche (décalés de +/- 150 degrés)
                angle_deg_1 = angle + math.radians(150)
                angle_deg_2 = angle + math.radians(210) 

                # Côté 1 de la flèche
                ax1 = x1_live_scaled_vec + ARROW_SIZE * math.cos(angle_deg_1)
                ay1 = y1_live_scaled_vec + ARROW_SIZE * math.sin(angle_deg_1)

                # Côté 2 de la flèche
                ax2 = x1_live_scaled_vec + ARROW_SIZE * math.cos(angle_deg_2)
                ay2 = y1_live_scaled_vec + ARROW_SIZE * math.sin(angle_deg_2)

                painter.setPen(pen_vector)
                painter.drawLine(int(x1_live_scaled_vec), int(y1_live_scaled_vec), int(ax1), int(ay1))
                painter.drawLine(int(x1_live_scaled_vec), int(y1_live_scaled_vec), int(ax2), int(ay2))


    def toggle_live_video(self):
        """Démarre ou arrête le QTimer pour le flux vidéo live."""
        if self.is_paused:
            self.timer.start(30)
            self.pause_button.setText("Pause")
            self.pause_button.setObjectName("pause_button")
            self.pause_button.setStyleSheet(self._get_qstyle()) 
            print("▶️ Flux vidéo live relancé.")
        else:
            self.timer.stop()
            self.pause_button.setText("Lecture")
            self.pause_button.setObjectName("play_button")
            self.pause_button.setStyleSheet(self._get_qstyle()) 
            print("⏸️ Flux vidéo live mis en pause.")
        
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            # Assure la mise à jour de la dernière image si on met en pause
            self.update_live_image()


    # --- Mise à jour de la vidéo live ---
    def update_live_image(self): 
        if self.timer.isActive() or self.is_paused: 
            
            frame = get_live_image()
            if frame is not None:
                centers_roi, _ = detect_spots_cells(frame, self.grid_cells)
                
                x_offset = self.coords.get('x_min', 0)
                y_offset = self.coords.get('y_min', 0)
                
                centers_full_ref = []
                for cx_roi, cy_roi in centers_roi:
                    centers_full_ref.append((cx_roi + x_offset, cy_roi + y_offset))
                
                
                cells = compute_cells_from_grid_ref(centers_full_ref, self.vertical_lines, self.horizontal_lines)
                vectors = compare_grid_cells_and_compute_vectors(cells)
                
                self.current_vectors = vectors 
                
                # Le print du DeltaXY Moyen a été retiré, comme demandé.
                
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image)
                
                pixmap = pixmap.scaled(self.live_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

                painter = QPainter(pixmap) 

                if self.show_grid and self.grid_pixmap and not self.grid_pixmap.isNull():
                    painter.drawPixmap(0, 0, self.grid_pixmap)
                
                self.draw_vectors(painter) 

                painter.end()

                self.live_label.setPixmap(pixmap)


    # --- Mise à jour de l'image de référence ---
    def update_reference_image(self):
        ref_image = cv2.imread('image_reference_centre.png')
        if ref_image is not None:
            if len(ref_image.shape) == 2:
                ref_image = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
            
            ref_image_resized = cv2.resize(ref_image, 
                                             (self.REF_DISPLAY_WIDTH, self.REF_DISPLAY_HEIGHT), 
                                             interpolation=cv2.INTER_LANCZOS4)
            
            h, w, ch = ref_image_resized.shape
            bytes_per_line = ch * w
            qt_image = QImage(ref_image_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
            
            pixmap = QPixmap.fromImage(qt_image)
            self.ref_label.setPixmap(pixmap)

    # --- Acquisition d'une nouvelle image de référence ---
    def acquire_new_reference_image(self):
        print("🔵 Acquisition d’une nouvelle image de référence...")
        self.acquire_button.setEnabled(False) 
        QApplication.processEvents() 
        
        was_running = self.timer.isActive()
        if was_running:
             self.timer.stop()
        
        try:
            capture_and_save_image() 
            process_and_save_images('image_reference.png')
            
            self.coords = assign_coordinates_from_file('reference.txt')
            if self.coords:
                set_camera_roi(self.coords['x_min'], self.coords['y_min'],
                               self.coords['x_max'] - self.coords['x_min'],
                               self.coords['y_max'] - self.coords['y_min'])
                print(f"🎯 Nouveau ROI appliqué : {self.coords}")
            else:
                print("❌ Échec de la détection de nouvelles coordonnées.")
                
            self.vertical_lines, self.horizontal_lines = read_grid_from_reference('reference.txt')
            self.grid_cells = read_grid_cells_from_reference('reference.txt') 
            
            self.create_grid_pixmap(self.live_label.width(), self.live_label.height())
            self.update_reference_image()

        except Exception as e:
            print(f"❌ Erreur lors de l'acquisition de référence : {e}")

        finally:
            self.acquire_button.setEnabled(True) 
            if was_running:
                 self.timer.start(30)


    # --- Mise à jour de l'exposition ---
    def update_exposure(self, value_us):
        """Met à jour l'exposition de la caméra en microsecondes (µs)."""
        try:
            set_camera_exposure(value_us)
            print(f"✅ Exposition réglée sur {value_us / 1e6:.6f} s ({value_us} µs)")
        except Exception as e:
            print(f"❌ Erreur exposition : {e}")
            
    # --- Appliquer zoom ROI (inchangée) ---
    def apply_zoom_roi(self):
        self.coords = assign_coordinates_from_file('reference.txt')
        if self.coords:
            set_camera_roi(self.coords['x_min'], self.coords['y_min'],
                           self.coords['x_max'] - self.coords['x_min'],
                           self.coords['y_max'] - self.coords['y_min'])
            print(f"🎯 ROI de Zoom appliqué : {self.coords}")
            
            self.vertical_lines, self.horizontal_lines = read_grid_from_reference('reference.txt')
            self.grid_cells = read_grid_cells_from_reference('reference.txt') 
            self.create_grid_pixmap(self.live_label.width(), self.live_label.height())


if __name__ == "__main__":
    # Assurez-vous d'avoir une classe QDoubleSpinBox disponible si vous l'utilisez
    from PyQt5.QtWidgets import QDoubleSpinBox 
    
    app = QApplication(sys.argv)
    window = LiveReferenceWindow()
    sys.exit(app.exec_())