import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

import pytest


@pytest.fixture(scope="session")
def data():
    import data_loader as dl
    panel, sti = dl.load_panel()
    return panel, sti
