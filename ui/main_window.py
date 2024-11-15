# File: Application/ui/main_window.py


"""
This file defines the MainWindow class, which serves as the primary user interface
for the SOEN Model Monitor application. It creates a multi-tabbed window that allows
users to interact with and monitor different neural network models, for
MNIST and Two Moons datasets.

The MainWindow initialises separate SOEN models for each dataset, loads the required
data, and sets up the user interface with tabs for training and visualisation.

"""
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from .mnist_training_tab import TrainingTab
from .two_moons_training_tab import TwoMoonsTrainingTab
from config import WINDOW_WIDTH, WINDOW_HEIGHT
from model.soen_model import SOENModel
from model_config_files.mnist_config import MNISTConfig
from model_config_files.two_moons_config import TwoMoonsConfig
from utils.data_loader import load_mnist_data, load_two_moons_data
from utils.two_moons_utils import prepare_two_moons_data
import logging

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOEN Model Monitor")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Create separate models for MNIST and Two Moons
        logging.info("Initialising SOEN model for MNIST...")
        self.mnist_model = SOENModel(MNISTConfig())
        logging.info("Initialising SOEN model for Two Moons...")
        self.two_moons_model = SOENModel(TwoMoonsConfig())

        self.mnist_train_loader, self.mnist_val_loader, self.mnist_test_loader = load_mnist_data()
        
        # Prepare Two Moons data
        X, y, self.two_moons_scaler = prepare_two_moons_data()
        self.two_moons_train_loader, self.two_moons_val_loader = load_two_moons_data(X, y, self.two_moons_scaler)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create and add tabs with separate models and data loaders
        self.mnist_training_tab = TrainingTab(self.mnist_model, self.mnist_train_loader, self.mnist_val_loader)
        self.two_moons_training_tab = TwoMoonsTrainingTab(self.two_moons_model, self.two_moons_train_loader, self.two_moons_val_loader, self.two_moons_scaler)
        
        

        self.tab_widget.addTab(self.mnist_training_tab, "MNIST Training")
        self.tab_widget.addTab(self.two_moons_training_tab, "Two Moons Training")
        

        # Set the window to stay on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)