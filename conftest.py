# content of conftest.py
import pytest
import yaml

def pytest_addoption(parser):
    parser.addoption("--lecroy_dir", action="store", default="type1",
        help="lecroy_dir: directory where LeCroy CSV test files can be found")

@pytest.fixture
def lecroy_dir(request):
    return request.config.getoption("--lecroy_dir")

def pytest_generate_tests(metafunc):
    if 'test' in metafunc.fixturenames:
        with open('main_test_list.yaml', 'r') as f:
            Iterations = yaml.load(f)
            metafunc.parametrize('test', Iterations)