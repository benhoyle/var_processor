"""Test routines for GUI.

Run: pytest --cov=src
"""

import numpy as np
import tkinter as tk
from src.visualisers.abstract_gui import AbstractGUI, ImagePanels


class Sensor:
    """Mock class for sensor."""
    def __init__(self):
        rng = np.random.default_rng()

    def start(self):
        pass

    def stop(self):
        pass


def test_abstract_gui():
    """Test abstract GUI abject."""
    # Create dummy sensor object
    sensor = Sensor()
    sensor.started = False
    gui = AbstractGUI(sensor)
    gui.run()


def test_add_sensor():
    assert False


def test_update():
    assert False


def test_run():
    assert False


def test_quit_():
    assert False


class TestImagePanels:

    def test_create(self):
        # Create a test parent window
        root = tk.Tk()
        self.panels = ImagePanels(root)
        root.destroy()

    def test_update(self):
        # Create fake window
        root = tk.Tk()
        self.panels = ImagePanels(root)
        # Initialise numpy RNG
        rng = np.random.default_rng()
        # Create a list of dummy random images
        images = [rng.integers(0, 255, size=(128, 128), dtype=np.uint8) for i in range(4)]
        self.panels.update(images)
        root.destroy()

    def test_panels_gui(self):
        """Test panels within an abstract GUI."""
        sensor = Sensor()
        sensor.started = False
        gui = AbstractGUI(sensor)
        gui.panels = [ImagePanels(gui.window)]
        # Initialise numpy RNG
        rng = np.random.default_rng()
        # Create a list of dummy random images
        images = [rng.integers(0, 255, size=(128, 128), dtype=np.uint8) for i in range(4)]
        gui.panels[0].update(images)
        gui.run()
