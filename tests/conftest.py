import pytest

from src.common.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    s = get_spark("audiencegraph-tests")
    yield s
    s.stop()
