# FILENAME: main.py

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from model.soen_model import SOENModel
from model.model_config import SOENConfig
import logging
import torch


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def main():
    
    """
    Entry point of the SOEN Monitor application.
    Sets up the data, model, and launches the main window. 
    """

    setup_logging()
    logging.info("Starting application...")

    # Set up and run the application
    logging.info("Setting up main window...")
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    logging.info("Main window displayed. Starting event loop...")
    sys.exit(app.exec_())

if __name__ == "__main__":

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    main()