from jax.config import config
config.update("jax_enable_x64", True)
from .curve import RotatedCurve
from .curverzfourier import *
from .curvexyzfourier import *
from .objectives import *
from .config import *
from .biotsavart import *
from .surface import *
from .surfacerzfourier import *
from .surfacexyzfourier import *
# from .magnetic_surface import *
from .magnetic_surface_leastsquares import *
