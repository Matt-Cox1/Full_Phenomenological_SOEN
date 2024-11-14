# File: Application/ui/main_window.py


"""
This file defines the MainWindow class, which serves as the primary user interface
for the SOEN Model Monitor application. It creates a multi-tabbed window that allows
users to interact with and monitor different neural network models, for
MNIST, Two Moons, and Spoken Digits datasets.

The MainWindow initialises separate SOEN models for each dataset, loads the required
data, and sets up the user interface with tabs for training and visualisation.

"""
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from .mnist_training_tab import TrainingTab
from .two_moons_training_tab import TwoMoonsTrainingTab
from .spoken_digit_tab import SpokenDigitTrainingTab
from config import WINDOW_WIDTH, WINDOW_HEIGHT
from model.soen_model import SOENModel
from model_config_files.mnist_config import MNISTConfig
from model_config_files.two_moons_config import TwoMoonsConfig
from model_config_files.spokendigit_rnn_config import SpokenDigitRNNConfig
from utils.data_loader import load_mnist_data, load_two_moons_data, load_audio_data
from utils.two_moons_utils import prepare_two_moons_data
from utils.soen_model_utils import visualise_soen_network
import logging

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOEN Model Monitor")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Create separate models for all datasets
        logging.info("Initialising SOEN model for Spoken Digits...")
        self.spoken_digit_model = SOENModel(SpokenDigitRNNConfig())

        # Visualize the network structure
        # visualise_soen_network(
        #     self.spoken_digit_model,
        #     dpi=300,
        #     show_labels=True,
        #     dark_mode=False,
        #     node_size=50,
        #     figsize=(18, 8),
        #     line_thickness=0.5,
        #     legend_fontsize=10,
        #     layout='linear',
        #     show_legend=True
        # )

        # Load Spoken Digit data
        self.spoken_digit_train_loader, self.spoken_digit_val_loader, self.spoken_digit_test_loader = load_audio_data()

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create and add tabs with separate models and data loaders
        self.spoken_digit_training_tab = SpokenDigitTrainingTab(self.spoken_digit_model, self.spoken_digit_train_loader, self.spoken_digit_val_loader)

        self.tab_widget.addTab(self.spoken_digit_training_tab, "Spoken Digits Training")

        # Set the window to stay on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)