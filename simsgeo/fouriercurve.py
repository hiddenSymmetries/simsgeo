from .curve import Curve
import simsgeopp as sgpp

class FourierCurve(sgpp.FourierCurve, Curve):

    def __init__(self, *args):
        sgpp.FourierCurve.__init__(self, *args)
