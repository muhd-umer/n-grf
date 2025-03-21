# engine/tests/conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--data_path",
        action="store",
        default="datasets/outputs/conf_16x2_414u_5.0ghz_sbrRT_sc104.mat",
        help="Path to the dataset file",
    )


@pytest.fixture(scope="session")
def data_path(request):
    return request.config.getoption("--data_path")
