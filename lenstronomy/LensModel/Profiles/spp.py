__author__ = "sibirrer"


import numpy as np
import scipy.special as special
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["SPP"]


class SPP(LensProfileBase):
    """Class for Spherical Power Law Potential (SPP).

    .. math::
        \\psi(x, y) = \\frac{2 E^2}{\\eta^2} \\left( \\frac{p^2}{E^2} \\right)^{\\frac{\\eta}{2}}

    where :math:`\\theta_E` is the Einstein radius,
    :math:`\\gamma` is the power-law slope of the mass distribution (with :math:`\\gamma < 2`),
    :math:`\\eta = -\\gamma` + 3 is the power-law exponent transformation,
    :math:`E` is an effective Einstein radius based on :math:`\\theta_E` and :math:`\\gamma`, defined as

    .. math::
        E = \\frac{\\theta_E}{\\left( \\frac{3 - \\gamma}{2} \\right)^{\\frac{1}{1 - \\gamma}}},

    and :math:`p^2` is the radius squared from the center, calculated from the coordinates :math:`x` and :math:`y`.

    For a full mathematical derivation see Suyu (2012), https://ui.adsabs.harvard.edu/abs/2012MNRAS.426..868S/abstract.
    """

    param_names = ["theta_E", "gamma", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "center_x": 100,
        "center_y": 100,
    }

    def function(self, x, y, theta_E, gamma, center_x=0, center_y=0):
        """
        :param x: set of x-coordinates
        :type x: array of size (n)
        :param y: set of y-coordinates
        :type y: array of size (n)
        :param theta_E: Einstein radius of lens
        :type theta_E: float.
        :param gamma: power law slope of mass profile
        :type gamma: <2 float
        :returns:  function
        :raises: AttributeError, KeyError
        """
        gamma = self._gamma_limit(gamma)

        x_ = x - center_x
        y_ = y - center_y
        E = theta_E / ((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3

        p2 = x_**2 + y_**2
        return 2 * E**2 / eta**2 * ((p2) / E**2) ** (eta / 2)

    def derivatives(self, x, y, theta_E, gamma, center_x=0.0, center_y=0.0):
        """
        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
          
        gamma = self._gamma_limit(gamma)

        xt1 = x - center_x
        xt2 = y - center_y

        r2 = xt1 * xt1 + xt2 * xt2
        a = np.maximum(r2, 0.000001)
        r = np.sqrt(a)
        alpha = theta_E * (r2 / theta_E**2) ** (1 - gamma / 2.0)
        fac = alpha / r
        f_x = fac * xt1
        f_y = fac * xt2
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, center_x=0.0, center_y=0.0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
            
        gamma = self._gamma_limit(gamma)
        xt1 = x - center_x
        xt2 = y - center_y
        E = theta_E / ((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3.0

        P2 = xt1**2 + xt2**2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001, P2)
        else:
            a = np.empty_like(P2)
            p2 = P2[P2 > 0]  # in the SIS regime
            a[P2 == 0] = 0.000001
            a[P2 > 0] = p2

        kappa = (
            1.0
            / eta
            * (a / E**2) ** (eta / 2 - 1)
            * ((eta - 2) * (xt1**2 + xt2**2) / a + (1 + 1))
        )
        gamma1 = (
            1.0
            / eta
            * (a / E**2) ** (eta / 2 - 1)
            * ((eta / 2 - 1) * (2 * xt1**2 - 2 * xt2**2) / a)
        )
        gamma2 = (
            4 * xt1 * xt2 * (1.0 / 2 - 1 / eta) * (a / E**2) ** (eta / 2 - 2) / E**2
        )

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def rho2theta(rho0, gamma):
        """Converts 3d density into 2d projected density parameter.

        :param rho0: initial density parameter
        :param gamma: power-law slope
        :return: 2d projected density parameter
        """

        fac = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * 2
            / (3 - gamma)
            * rho0
        )

        theta_E = fac ** (1.0 / (gamma - 1))
        return theta_E

    @staticmethod
    def theta2rho(theta_E, gamma):
        """Converts projected density parameter (in units of deflection) into 3d density
        parameter.

        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :return: 3d density parameter
        """

        fac1 = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * 2
            / (3 - gamma)
        )
        fac2 = theta_E ** (gamma - 1)
        rho0 = fac2 / fac1
        return rho0

    @staticmethod
    def mass_3d(r, rho0, gamma):
        """Mass enclosed in a 3d sphere of radius r.

        :param r: radius of the sphere
        :param rho0: 3d density parameter
        :param gamma: power-law slope
        :return: mass enclosed within radius r
        """

        mass_3d = 4 * np.pi * rho0 / (-gamma + 3) * r ** (-gamma + 3)
        return mass_3d

    def mass_3d_lens(self, r, theta_E, gamma):
        """
        Computes the mass enclosed in a 3d sphere of radius r using lens model parameters.

        :param r: radius of the sphere
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :return: mass enclosed within radius r
        """

        rho0 = self.theta2rho(theta_E, gamma)
        return self.mass_3d(r, rho0, gamma)

    def mass_2d(self, r, rho0, gamma):
        """Mass enclosed in a projected 2d sphere of radius r.

        :param r: projected radius
        :param rho0: 3d density parameter
        :param gamma: power-law slope
        :return: mass enclosed within projected radius r
        """

        alpha = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * r ** (2 - gamma)
            / (3 - gamma)
            * 2
            * rho0
        )
        mass_2d = alpha * r * np.pi
        return mass_2d

    def mass_2d_lens(self, r, theta_E, gamma):
        """
        mass enclosed in 2d sphere of radius r

        :param r: projected radius
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :return: mass enclosed within projected radius r
        """

        rho0 = self.theta2rho(theta_E, gamma)
        return self.mass_2d(r, rho0, gamma)

    def grav_pot(self, x, y, rho0, gamma, center_x=0, center_y=0):
        """Gravitational potential (modulo 4 pi G and rho0 in appropriate units).

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param rho0: 3d density parameter
        :param gamma: power-law slope
        :param center_x: x-coordinate of profile center
        :param center_y: y-coordinate of profile center
        :return: gravitational potential
        """

        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        mass_3d = self.mass_3d(r, rho0, gamma)
        pot = mass_3d / r
        return pot

    @staticmethod
    def density(r, rho0, gamma):
        """Computes the density.

        :param r: radius
        :param rho0: 3d density parameter
        :param gamma: power-law slope
        :return: density at radius r
        """
        rho = rho0 / r**gamma
        return rho

    def density_lens(self, r, theta_E, gamma):
        """Computes the density at 3d radius r given lens model parameterization.

        The integral in projected in units of angles (i.e. arc seconds) results in the
        convergence quantity.

        :param r: radius
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        """

        rho0 = self.theta2rho(theta_E, gamma)
        return self.density(r, rho0, gamma)

    @staticmethod
    def density_2d(x, y, rho0, gamma, center_x=0, center_y=0):
        """Projected density.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param rho0: 3d density parameter
        :param gamma: power-law slope
        :param center_x: x-coordinate of profile center
        :param center_y: y-coordinate of profile center
        :return: projected density
        """
    
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        sigma = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * r ** (1 - gamma)
            * rho0
        )
        return sigma

    @staticmethod
    def _gamma_limit(gamma):
        """Limits the power-law slope to certain bounds.

        :param gamma: power-law slope
        :return: bounded power-law slope
        """

        if gamma > 2.:
            gamma = 2.

        return gamma
