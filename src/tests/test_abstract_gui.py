"""Test routines for GUI.

Run: pytest --cov=src
"""

from src.visualisers.abstract_gui import AbstractGUI


class Sensor:
    """Mock class for sensor."""
    pass

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
