from pathlib import Path
import pickle
import os
import uuid

class WrappedData:
    def __init__(self, filename):
        self._filename = filename

    def unwrap(self):
        with open(self._filename, "rb") as f:
            return pickle.load(f)

    def cleanup(self):
        if os.path.exists(self._filename):
            os.remove(self._filename)

class DataStore:
    def __init__(self, data_dir):
        self._data_dir = Path(data_dir)
        
    def wrap(self, data):
        if isinstance(data, WrappedData):
            return data
        else:
            object_id = uuid.uuid4().hex
            filename = self._data_dir / f"{object_id}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            return WrappedData(filename)

