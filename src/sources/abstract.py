"""Abstract Sources."""

import threading


class SensorSource:
    """Abstract object for a sensory modality."""

    def __init__(self):
        """Initialise object."""
        self.started = False
        self.thread = None

    def start(self):
        """Start capture source."""
        if self.started:
            print('[!] Asynchroneous capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(
            target=self.update,
            args=()
        )
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        """Update data."""
        pass

    def read(self):
        """Read data."""
        pass

    def stop(self):
        """Stop daemon."""
        self.started = False
        self.thread.join()


class CombinedSource:
    """Object to combine multiple modalities."""

    def __init__(self):
        """Initialise."""
        self.sources = dict()
        self.started = False

    def add_source(self, source, name=None):
        """Add a source object.

        source is a derived class from SensorSource
        name is an optional string name.
        """
        if not name:
            name = source.__class__.__name__
        self.sources[name] = source

    def start(self):
        """Start all sources."""
        for _, source in self.sources.items():
            source.start()
        self.started = True

    def read(self):
        """Read from all sources.

        return as dict of tuples.
        """
        data = dict()
        for name, source in self.sources.items():
            data[name] = source.read()[1]
        return data

    def stop(self):
        """Stop all sources."""
        for _, source in self.sources.items():
            source.stop()
        self.started = False

    def __del__(self):
        """Extra code to close camera."""
        for _, source in self.sources.items():
            if source.__class__.__name__ == "VideoSource":
                source.cap.release()

    def __enter__(self):
        """Enter - Dummy."""
        return self

    def __exit__(self, exec_type, exc_value, traceback):
        """Extra code to close camera."""
        for _, source in self.sources.items():
            if source.__class__.__name__ == "VideoSource":
                source.cap.release()
