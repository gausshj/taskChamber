import pytest


@pytest.fixture
def anyio_backend() -> str:
    """The locked environment includes AnyIO's pytest plugin but not Trio."""

    return "asyncio"
