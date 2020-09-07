import numpy as np

class Curve(object):

    def plot(self, ax=None, show=True, plot_derivative=False, closed_loop=True, color=None, linestyle=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        gamma = self.gamma()
        dgamma_by_dphi = self.dgamma_by_dphi()[0, :,:]
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        def rep(data):
            if closed_loop:
                return np.concatenate((data, [data[0]]))
            else:
                return data
        ax.plot(rep(gamma[:, 0]), rep(gamma[:, 1]), rep(gamma[:, 2]), color=color, linestyle=linestyle)
        if plot_derivative:
            ax.quiver(rep(gamma[:, 0]), rep(gamma[:, 1]), rep(gamma[:, 2]), 0.1 * rep(dgamma_by_dphi[:, 0]), 0.1 * rep(dgamma_by_dphi[:, 1]), 0.1 * rep(dgamma_by_dphi[:, 2]), arrow_length_ratio=0.1, color="r")
        if show:
            plt.show()
        return ax

