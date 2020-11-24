import sys

try:
    from skbuild import setup
except ImportError:
    print('Please update pip, you need pip 10 or greater,\n'
          ' or you need to install the PEP 518 requirements in pyproject.toml yourself', file=sys.stderr)
    raise

setup(
    name='simsgeo',
    version="0.0.1",
    description="A collection of geometric objects useful for Stellarator optimisation",
    author='Florian Wechsung',
    packages=['simsgeo', 'simsgeopp'],
    package_dir={'simsgeo': 'simsgeo', 'simsgeopp': 'simsgeopp'},
    cmake_install_dir='simsgeopp'
)
