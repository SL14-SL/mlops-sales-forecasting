from importlib import import_module


def test_prefect_server_routes_can_be_imported() -> None:
    server = import_module("prefect.server.api.server")

    assert callable(server.create_app)