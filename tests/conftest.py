import pytest

from modlab.server import create_app


@pytest.fixture(scope="session")
def app():
    return create_app({"TESTING": True})


@pytest.fixture()
def client(app):
    return app.test_client()
