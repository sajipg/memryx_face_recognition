import os
import sys
import glob
import cv2
import numpy as np

from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog, QDialog,
                               QMessageBox, QFileDialog)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex
from PySide6.QtGui import QAction
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidgetAction, QSizePolicy
from PySide6.QtWidgets import QToolButton   
from PySide6.QtCore import QSize
import time
from pathlib import Path

from modules.capture import CaptureThread, CaptureConfigDialog, VIDEO_CONFIG
from modules.compositor import Compositor, CompositorConfigPopup
from modules.viewer import FrameViewer
from modules.database import FaceDatabase, DatabaseViewerWidget
from modules.tracker import FaceTracker

class face_recognition(QMainWindow):
    def __init__(self, video_path='/dev/video0', video_config=None):
        super().__init__()
        # Set window title and icon
        self.setWindowTitle("AetherHive AI Face Identifier")
        self.setWindowIcon(QIcon("../assets/aetherhive-logo.ico"))
        
        # Create the video display.
        self.viewer = FrameViewer()

        # Set up video capture and processing.
        self.capture_thread = CaptureThread(video_path, video_config)
        self.face_database = FaceDatabase()
        self.database_viewer = DatabaseViewerWidget(self.face_database)
        self.tracker = FaceTracker(self.face_database)
        self.compositor = Compositor(self.tracker)
        self.compositor.set_paused(self.capture_thread.pause)

        # Create a button to open the compositor config popup.
        self.config_popup_button = QPushButton("Display Config", self)
        self.config_popup_button.setFixedSize(200, 30)  # width, height
        self.config_popup_button.clicked.connect(self.open_compositor_config)

        self.capture_control_button = QPushButton("Video Source Config", self)
        self.capture_control_button.setFixedSize(200, 30)  # width, height
        self.capture_control_button.clicked.connect(self.open_capture_config)

        # Wire signals.
        self.capture_thread.frame_ready.connect(self.tracker.detect)
        self.tracker.frame_ready.connect(self.compositor.draw)
        self.compositor.frame_ready.connect(self.viewer.update_frame)
        self.viewer.mouse_move.connect(self.compositor.update_mouse_pos)
        self.viewer.mouse_click.connect(self.handle_viewer_mouse_click)

        # Add click-to-pause functionality: clicking anywhere in the viewer toggles capture play/pause
        self.viewer.mouse_click.connect(self.toggle_capture_pause)

        self.setup_layout()

        self.tracker.start()
        self.capture_thread.start()
        self.timestamps = [0] * 30

        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(500)
        self.fps_timer.timeout.connect(self.poll_framerates)
        self.fps_timer.start()

        # Create a persistent instance for the config popup.
        self.config_popup = None
        
    def toggle_control_panel(self):
        if self.toggle_panel_action.isChecked():
            self.control_container.show()
            self.toggle_panel_action.setText("Toggle Control Panel")
        else:
            self.control_container.hide()
            self.toggle_panel_action.setText("Show Control Panel")
        
        
    def setup_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setMinimumSize(300, 200)

        # Main vertical layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ───────────── Title Bar ─────────────
        title_bar = QWidget()
        title_bar.setFixedHeight(50)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 0, 0)
        title_layout.setSpacing(10)
        title_layout.setAlignment(Qt.AlignLeft)

        logo_label = QLabel()
        logo_pixmap = QPixmap("../assets/aetherhive-logo.png")
        logo_label.setPixmap(logo_pixmap.scaledToWidth(30, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_layout.addWidget(logo_label)

        #title_text = QLabel("AetherHive Face Recognition")
        #title_text.setStyleSheet("font-size: 16px; font-weight: normal; margin: 0px; padding: 0px;")
        #title_text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        #title_layout.addWidget(title_text)

        title_bar.setLayout(title_layout)
        self.main_layout.addWidget(title_bar)

        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #ccc;")
        separator.setFixedHeight(2)
        self.main_layout.addWidget(separator)

        # ───────────── Menu Bar ─────────────
        menu_bar = self.menuBar()
        config_menu = menu_bar.addMenu("Config")
        about_menu = menu_bar.addMenu("About")

        about_action = QAction("Open About Popup", self)
        about_action.triggered.connect(self.show_about_popup)
        about_menu.addAction(about_action)

        self.toggle_panel_action = QAction("Hide Control Panel", self)
        self.toggle_panel_action.setCheckable(True)
        self.toggle_panel_action.setChecked(False)
        self.toggle_panel_action.triggered.connect(self.toggle_control_panel)
        config_menu.addAction(self.toggle_panel_action)

        

        # ───────────── Splitter ─────────────
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #ccc;
                width: 2px;
            }
        """)
        self.main_layout.addWidget(self.splitter)

        # ───────────── Framed Video Viewer ─────────────
        viewer_frame = QFrame()
        #viewer_frame.setFrameShape(QFrame.Box)
        #viewer_frame.setLineWidth(1)
        viewer_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;  /* Light grey border */
                border-radius: 2px;      /* Optional: slightly rounded corners */
                background-color: transparent;
            }
        """)

        viewer_layout = QVBoxLayout(viewer_frame)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self.viewer)

        self.splitter.addWidget(viewer_frame)

        # ───────────── Control Panel ─────────────
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(300)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(10, 10, 10, 10)
        self.control_layout.setSpacing(10)

        self.control_layout.addWidget(self.capture_control_button)
        self.control_layout.addWidget(self.config_popup_button)
        self.control_layout.addWidget(self.database_viewer)

        # Create a frame to wrap the control panel
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;  /* Medium grey border */
                border-radius: 4px;
                background-color: transparent;
            }
        """)

        frame_layout = QVBoxLayout(control_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame_layout.addWidget(self.control_panel)

        # Replace control_container with the framed version
        self.control_container = control_frame

        self.splitter.addWidget(self.control_container)
        self.control_container.hide()

        # Stretch factors
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        

    def show_about_popup(self):
        popup = AboutPopup(self)
        popup.exec()
    
    def handle_submit(self):
        text = self.text_input.text()
        print(f"Feedback submitted: {text}")
        self.accept()


    def open_compositor_config(self):
        if self.config_popup is None:
            self.config_popup = CompositorConfigPopup(self.compositor, self)
        self.config_popup.show()
        self.config_popup.raise_()

    def open_capture_config(self):
        current_resolution = "2k"
        for res in ["1080p", "2k", "4k"]:
            if VIDEO_CONFIG.get(res) == self.capture_thread.video_config:
                current_resolution = res
                break

        dialog = CaptureConfigDialog(self.capture_thread.video_source, current_resolution, self)
        if dialog.exec() == QDialog.Accepted:
            new_video_path, new_resolution = dialog.get_configuration()
            print(f"Applying new capture configuration: {new_video_path}, {new_resolution}")
            self.capture_thread.stop()
            self.capture_thread.wait()
            new_config = VIDEO_CONFIG.get(new_resolution, self.capture_thread.video_config)
            self.capture_thread = CaptureThread(new_video_path, new_config)
            self.capture_thread.frame_ready.connect(self.tracker.detect)
            self.capture_thread.start()

    def handle_viewer_mouse_click(self, mouse_pos):
        # Existing face-capture logic
        if mouse_pos is None:
            return
        
        tracker_frame = np.copy(self.tracker.current_frame.image)
        tracker_objects = self.tracker.get_activated_tracker_objects()

        found = False 
        mouse_x, mouse_y = mouse_pos
        for obj in tracker_objects:
            (left, top, right, bottom) = obj.bbox
            if left <= mouse_x <= right and top <= mouse_y <= bottom:
                width = right - left
                height = bottom - top
                margin = 10
                bbox_size = max(width, height) + 2 * margin
                center_x, center_y = left + width // 2, top + height // 2
                x_start = max(0, center_x - bbox_size // 2)
                x_end = min(tracker_frame.shape[1], center_x + bbox_size // 2)
                y_start = max(0, center_y - bbox_size // 2)
                y_end = min(tracker_frame.shape[0], center_y + bbox_size // 2)
                cropped_frame = tracker_frame[y_start:y_end, x_start:x_end]

                profile_path = self.database_viewer.get_selected_directory()
                if not profile_path:
                    if obj.name == 'Unknown':
                        new_profile = self.database_viewer.add_profile()
                        profile_path = os.path.join(self.database_viewer.db_path, new_profile)
                    else:
                        profile_path = os.path.join(self.database_viewer.db_path, obj.name)

                if os.path.exists(profile_path) and Path(profile_path) != Path(self.database_viewer.db_path):
                    i = 0
                    while os.path.exists(os.path.join(profile_path, f"{i}.jpg")):
                        i += 1
                    filename = os.path.join(profile_path, f"{i}.jpg")
                    print(f'Saving image to {filename}')
                    cv2.imwrite(filename, cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
                    self.database_viewer.load_profiles()
                    self.face_database.add_to_database(obj.embedding, filename)
                found = True
                break

    def toggle_capture_pause(self, pos):
            """Toggle play/pause of the capture thread when the viewer is clicked, 
            but only if the click is NOT over a face."""
            if pos is None:
                return

            mouse_x, mouse_y = pos
            # If click is on any active tracked face, do not toggle pause.
            for obj in self.tracker.get_activated_tracker_objects():
                left, top, right, bottom = obj.bbox
                if left <= mouse_x <= right and top <= mouse_y <= bottom:
                    return  # clicked on a face; preserve current paused state

            # Click was not on a face: toggle pause/play.
            self.capture_thread.toggle_play()
            state = "paused" if self.capture_thread.pause else "running"
            print(f"Capture thread {state}")
            self.compositor.set_paused(self.capture_thread.pause)

    def poll_framerates(self):
        # If paused, redraw the last known frame so the overlay shows.
        if self.capture_thread.pause:
            if hasattr(self.tracker, "current_frame") and self.tracker.current_frame is not None:
                frame = np.copy(self.tracker.current_frame.image)
                self.compositor.draw(frame)

    def closeEvent(self, event):
        self.capture_thread.stop()
        self.capture_thread.wait()
        self.tracker.stop()
        self.fps_timer.stop()
        super().closeEvent(event)
        

class AboutPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About AetherHive")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout(self)

        label = QLabel("Enter your feedback or comment:")
        layout.addWidget(label)

        self.text_input = QLineEdit()
        layout.addWidget(self.text_input)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.handle_submit)
        layout.addWidget(submit_button)

    def handle_submit(self):
        feedback = self.text_input.text()
        print(f"Feedback submitted: {feedback}")
        self.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = "/dev/video0"  # Update this path as needed.
    streams = sorted(glob.glob('/dev/video*'))

    if not streams:
        video_path = '../assets/mx-logo.png'
    else:
        video_path = streams[0] 

    player =face_recognition(video_path, VIDEO_CONFIG['2k'])
    player.resize(1200, 800)
    player.show()
    sys.exit(app.exec())

