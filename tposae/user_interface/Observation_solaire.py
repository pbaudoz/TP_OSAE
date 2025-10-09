from catkit2 import TestbedProxy
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QSplitter, QPushButton, QFileDialog, QLineEdit, QMenu, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
import time
from astropy.io import fits

def numpy_to_qimage(np_array: np.ndarray) -> QImage:
    """Safely convert a 2D uint8 NumPy array to QImage."""
    if np_array.dtype != np.uint8:
        raise ValueError("Only uint8 arrays supported for grayscale QImage.")
    np_array = np.ascontiguousarray(np_array)
    height, width = np_array.shape
    return QImage(np_array.data, width, height, width, QImage.Format_Grayscale8).copy()


class ClickableLabel(QLabel):
    """QLabel that emits click coordinates and handles drag events."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.click_callback = None
        self.dragging = False
        self.last_pos = QPoint()
        self.drag_rect = QRect()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
            self.drag_rect = QRect(self.last_pos, self.last_pos)
            if self.click_callback:
                self.click_callback(event.pos())

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.drag_rect.setBottomRight(event.pos())
            self.update()
            if self.click_callback:
                self.click_callback(self.drag_rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            if self.click_callback:
                self.click_callback(self.drag_rect)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        # No rectangle drawing here anymore


class LiveImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Image Stream Viewer with Zoom")

        # Simulation parameters
        self.img_width = 1200
        self.img_height = 800
        self.noise_level = 50
        self.zoom_size = 50  # Half-size of zoom window
        self.zoom_rect = QRect(200, 200, self.zoom_size * 2, self.zoom_size * 2)  # Default zoom rectangle
        self.image_ready = False
        self.image_offset = QPoint(0, 0)  # Offset for the image in the window

        # Main image view
        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.click_callback = self.handle_click_and_drag

        # Zoom view
        self.zoom_label = QLabel()
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("border: 1px solid gray;")
    
        # Sliders
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(1023)
        self.min_slider.setValue(0)

        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setMinimum(1)
        self.max_slider.setMaximum(1024)
        self.max_slider.setValue(1024)

       # self.exp1_slider = QSlider(Qt.Horizontal)
       # self.exp1_slider.setMinimum(8)
       # self.exp1_slider.setMaximum(480000)
       # self.exp1_slider.setValue(1000)

        self.exp2_slider = QSlider(Qt.Horizontal)
        self.exp2_slider.setMinimum(8)
        self.exp2_slider.setMaximum(480000)
        self.exp2_slider.setValue(1000)
        self.current_choice = "Short"

        self.min_slider.valueChanged.connect(self.update_image_display)
        self.max_slider.valueChanged.connect(self.update_image_display)
        #self.exp1_slider.valueChanged.connect(self.update_image_display)
        self.exp2_slider.valueChanged.connect(self.update_image_display)

        # Slider labels and text boxes for manual input
        self.min_label = QLabel("Min:")
        self.min_value = QLineEdit(str(self.min_slider.value()))
        self.min_value.setValidator(None)  # Allow any value

        self.max_label = QLabel("Max:")
        self.max_value = QLineEdit(str(self.max_slider.value()))
        self.max_value.setValidator(None)  # Allow any value

        self.exp1_label = QLabel("Exp (short):")
        self.exp1_value = QLineEdit("1000")
        self.exp1_value.setValidator(None)  # Allow any value
        self.exp1_value.editingFinished.connect(self.update_slider_from_text)

        self.exp2_label = QLabel("Exp (long):")
        self.exp2_value = QLineEdit(str(self.exp2_slider.value()))
        self.exp2_value.setValidator(None)  # Allow any value

        self.min_value.textChanged.connect(self.update_slider_from_text)
        self.max_value.textChanged.connect(self.update_slider_from_text)
        #self.exp1_value.editingFinished.connect(self.update_slider_from_text)
        self.exp2_value.editingFinished.connect(self.update_slider_from_text)

        # Also update the text boxes when the slider moves
        self.min_slider.valueChanged.connect(self.update_min_value_text)
        self.max_slider.valueChanged.connect(self.update_max_value_text)
        #self.exp1_slider.valueChanged.connect(self.update_exp1_value_text)
        self.exp2_slider.valueChanged.connect(self.update_exp2_value_text)

        # Buttons
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)

        # Create a button
        self.choice_button = QPushButton("Choose an Option")

        # Create a menu and add actions
        menu = QMenu()
        menu.addAction("Short Exposure image", lambda: self.choice_selected("Short Exposure image"))
        menu.addAction("Long Exposure image", lambda: self.choice_selected("Long Exposure image"))
        menu.addAction("Mixed image", lambda: self.choice_selected("Mixed"))

        # Attach the menu to the button
        self.choice_button.setMenu(menu)

#        self.size_up_button = QPushButton("Increase Size")
#        self.size_up_button.clicked.connect(self.increase_zoom_size)

        # Layouts
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.min_label)
        controls_layout.addWidget(self.min_slider)
        controls_layout.addWidget(self.min_value)
        controls_layout.addWidget(self.max_label)
        controls_layout.addWidget(self.max_slider)
        controls_layout.addWidget(self.max_value)
        controls_layout.addWidget(self.save_button)
#        controls_layout.addWidget(self.size_up_button)

        controls_layout2 = QHBoxLayout()
        controls_layout2.addWidget(self.exp1_label)
        #controls_layout2.addWidget(self.exp1_slider)
        controls_layout2.addWidget(self.choice_button)
        controls_layout2.addWidget(self.exp1_value)
        controls_layout2.addWidget(self.exp2_label)
        controls_layout2.addWidget(self.exp2_slider)
        controls_layout2.addWidget(self.exp2_value)

        # Create splitter to separate full image and zoom image
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.image_label)
        splitter.addWidget(self.zoom_label)

        # Stretch factors: both images resize proportionally
        splitter.setStretchFactor(0, 3)  # Full image (left) will take 3 parts
        splitter.setStretchFactor(1, 1)  # Zoom image (right) will take 1 part

        # Set minimum sizes for the panels to avoid them becoming too small
        self.image_label.setMinimumSize(400, 300)  # Ensure the left image has a reasonable size
        self.zoom_label.setMinimumSize(200, 150)   # Ensure the zoom area has a reasonable size

        # Add the splitter to the main layout
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addLayout(controls_layout)
        layout.addLayout(controls_layout2)

        self.setLayout(layout)

        # Dummy image
        testbed = TestbedProxy('127.0.0.1',2345)
        camera_id = 'detector'
        self.cam = getattr(testbed, camera_id)
        self.cam.exposure_time = int(self.exp1_value.text())
        self.bit_depth = self.cam.config.get('bit_saturation',16)
        self.cam.start_acquisition()
        
        
        self.image_ready = True

        # Simulate live image updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image_stream)
        self.timer.start(100)  # Update every 100 ms (for example)

    def update_image_stream(self):
        """Simulate a live image stream."""
        print("Timer triggered")  # Add this
        self.cam.exposure_time = int(self.exp1_value.text())
        time.sleep(int(self.exp1_value.text())/1e6*3)
        frame_tmp = self.cam.images.get_next_frame()
        frame = self.cam.images.get_next_frame()
        print(self.cam.exposure_time)
        self.img_height = self.cam.height
        self.img_width = self.cam.width
        self.image1 = frame.data
        print(int(self.exp1_value.text()))
        #if (int(self.exp2_value.text()) > int(self.exp1_value.text())):
        self.cam.exposure_time = int(self.exp2_value.text())
        time.sleep(int(self.exp2_value.text())/1e6*3)
        frame_tmp = self.cam.images.get_next_frame() #dropping one frame
        frame2 = self.cam.images.get_next_frame()
        self.image2 = frame2.data
        print(self.cam.exposure_time)
        self.cam.exposure_time = int(self.exp1_value.text())
        if self.current_choice == "Short":    
            self.current_image = self.image1
        if self.current_choice == "Long":    
            self.current_image = self.image2
        if self.current_choice == "Mixed":    
            mask = self.image2 < 1020 # (2**self.bit_depth-5) #
            print([int(self.exp2_value.text()),int(self.exp1_value.text()),np.max(self.image1),np.max(self.image2)])
            self.image1[np.where(mask)] = self.frame2.data[np.where(mask)]
            self.current_image = self.image1
        self.update_image_display()

    def handle_click_and_drag(self, event_or_rect):
        """Handle click and drag to adjust zoom."""
        if isinstance(event_or_rect, QPoint):
            # Handle click to start zoom area
            self.zoom_rect = QRect(event_or_rect, event_or_rect)
        elif isinstance(event_or_rect, QRect):
            # Handle drag to define zoom area
            self.zoom_rect = event_or_rect

        self.update_image_display()

    def update_image_display(self):
        vmin = self.min_slider.value()
        vmax = self.max_slider.value()

        if vmin >= vmax:
            return

        img_scaled = self.get_scaled_image(self.current_image, vmin, vmax)
        qimg = numpy_to_qimage(img_scaled)

        # Display full image
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Get the offset between the image and the label's visible area
        self.image_offset = QPoint(
            (self.image_label.width() - pixmap.width()) // 2,
            (self.image_label.height() - pixmap.height()) // 2
        )

        # Check if pixmap is valid
        if pixmap.isNull():
            return

        # Draw zoom rectangle on the full image
        if not self.zoom_rect.isNull():
            painter = QPainter(pixmap)
            painter.setPen(Qt.red)

            # Mapping the rectangle to the scaled image size
            scale_x = pixmap.width() / self.img_width
            scale_y = pixmap.height() / self.img_height

            # Adjust the zoom rectangle's coordinates
            x1 = int((self.zoom_rect.left()-self.image_offset.x()))# * scale_x)
            y1 = int((self.zoom_rect.top()-self.image_offset.y()))# * scale_y)
            w = int(self.zoom_rect.width())# * scale_x)
            h = int(self.zoom_rect.height())# * scale_y)
           #### print(self.zoom_rect,x1,y1,w,h,self.image_offset.x(),self.image_offset.y(),scale_x, scale_y)
            painter.drawRect(x1, y1, w, h)
            painter.end()

        self.image_label.setPixmap(pixmap)

        # Show the zoomed-in image in the zoom view
        x1 = int((self.zoom_rect.left()-self.image_offset.x()) / scale_x)
        y1 = int((self.zoom_rect.top()-self.image_offset.y()) / scale_y)
        w = int(self.zoom_rect.width() / scale_x)
        h = int(self.zoom_rect.height() / scale_y)
      ####  print(x1,y1,w,h,self.image_offset.x(),self.image_offset.y())

        zoomed_image = self.get_scaled_image(self.current_image[y1:y1+h, x1:x1+w],vmin,vmax)
#        zoomed_image = self.current_image[self.zoom_rect.top():self.zoom_rect.bottom(), self.zoom_rect.left():self.zoom_rect.right()]
        #zoomed_image = np.clip(zoomed_image, vmin, vmax)
        zoom_qimg = numpy_to_qimage(zoomed_image)
        zoom_pixmap = QPixmap.fromImage(zoom_qimg)

        # Make the zoom image fill available space
        self.zoom_label.setPixmap(zoom_pixmap.scaled(
            self.zoom_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_scaled_image(self, image, vmin, vmax):
        """Scale the image to the desired range."""
        return ((np.clip(image, vmin, vmax)-vmin)/(vmax-vmin)*255).astype('uint8')

    def save_image(self):
        """Save the current image."""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.fits)")
        if filename:
            fits.writeto(filename,self.current_image)
            #self.current_image.save(filename)

#    def increase_zoom_size(self):
#        """Increase the zoom window size."""
#        self.zoom_size = min(self.zoom_size + 10, min(self.img_width, self.img_height) // 2)
#        self.update_image_display()

    def update_slider_from_text(self):
        """Update slider values from text input."""
        try:
            min_val = int(self.min_value.text())
            max_val = int(self.max_value.text())
            if 0 <= min_val < max_val <= self.bit_depth:
                self.min_slider.setValue(min_val)
                self.max_slider.setValue(max_val)

            exp1_val = int(self.exp1_value.text())
            if exp1_val < 20:
     #           time.sleep(0.5)
                exp1_val = int(self.exp1_value.text())
                exp1_val = 20   
                self.exp1_value.setText('20')
            if exp1_val > 480000:
     #           time.sleep(0.5)
                exp1_val = int(self.exp1_value.text())
                exp1_val = 480000
                self.exp1_value.setText('480000')
       #     self.exp1_slider.setValue(exp1_val) 
            exp2_val = int(self.exp2_value.text())
            if exp2_val < 20:
    #            time.sleep(0.2)
                exp2_val = 20   
                self.exp2_value.setText('20')
            if exp2_val > 480000:
    #            time.sleep(0.2)
                exp2_val = 480000
                self.exp2_value.setText('480000')
            self.exp2_slider.setValue(exp2_val) 

        except ValueError:
            pass

    def update_min_value_text(self):
        """Update min value text box when slider moves."""
        self.min_value.setText(str(self.min_slider.value()))

    def update_max_value_text(self):
        """Update max value text box when slider moves."""
        self.max_value.setText(str(self.max_slider.value()))

    #def update_exp1_value_text(self):
    #    """Update exp1 value text box when slider moves."""
    #    self.exp1_value.setText(str(self.exp1_slider.value()))

    def update_exp2_value_text(self):
        """Update exp2 value text box when slider moves."""
        self.exp2_value.setText(str(self.exp2_slider.value()))

    def choice_selected(self, choice):
        self.current_choice = choice
        self.choice_button.setText(choice)  # Update button label
        #QMessageBox.information(self, "Selection", f"You selected: {choice}")
    #printif choice = "Mixed":
 
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = LiveImageViewer()
    viewer.show()
    sys.exit(app.exec_())
