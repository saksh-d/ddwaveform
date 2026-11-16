from pathlib import Path
import pytest
from advisor.core import configure


def pytest_addoption(parser):
    parser.addoption("--scaling-artifacts", default=None, help="Explicit qualified scaling evidence directory")
    parser.addoption("--artifacts", default=None, help="Explicit provisioned local evidence directory")


def pytest_configure(config):
    # A clean clone has no ignored local/ directory yet; keep temp output local.
    if config.option.basetemp:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def arithmetic():
    configure()


@pytest.fixture(scope="session")
def artifacts(request):
    value = request.config.getoption("--artifacts")
    if value is None:
        pytest.skip("Local artifact checks omitted: pass --artifacts PATH after bootstrap")
    return Path(value).resolve()


@pytest.fixture(scope="session")
def scaling_artifacts(request):
    value=request.config.getoption("--scaling-artifacts")
    if value is None:
        pytest.skip("Scaling evidence tests omitted: pass --scaling-artifacts PATH")
    return Path(value).resolve()
