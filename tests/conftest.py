from pathlib import Path

import pytest
from qcdata.utils import water as water_struct


@pytest.fixture
def water():
    """Water Structure fixture"""
    return water_struct


@pytest.fixture
def test_data_dir():
    """Test data directory Path"""
    return Path(__file__).parent / "data"
