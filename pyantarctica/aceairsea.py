#
# Copyright 2017-2018 - Swiss Data Science Center (SDSC) and ACE-DATA/ASAID Project consortium.
# A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and
# Eidgenössische Technische Hochschule Zürich (ETHZ). Written within the scope
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import airsea


def LMoninObukov_bulk(U10N, SSHF, SLHF, STair):
    """
        Function to Monin Obukov Length scale as function of bulk heat fluxes, neutral wind speed and air temperature

        :param x: data series
        :param U10N: 10 meter neutral wind speed
        :param SSHF: surface sensible heat flux [W/m2]
        :param SLHF: surface latent heat flux [W/m2]
        :param STair: surface air temperature [C]
        
        :returns: LMO: Monin Obukov Length scale [m]
    """

    if type(U10N) != np.ndarray:
        U10N = np.array([U10N])
    if type(SSHF) != np.ndarray:
        SSHF = np.array([SSHF])
    if type(SLHF) != np.ndarray:
        SLHF = np.array([SLHF])
    if type(STair) != np.ndarray:
        STair = np.array([STair])

    Cp = airsea.constants.cp  # 1004.7 or 1005 # J/kg/K
    Levap = airsea.atmosphere.vapor(STair)  # ~2.5e+6 J/kg
    rho_air = airsea.atmosphere.air_dens(Ta=STair, rh=(STair * 0))
    vKarman = airsea.constants.kappa
    # van Karman constant
    grav = airsea.constants.g
    # const of gravitation

    wt = SSHF / rho_air / Cp
    wq = SLHF / rho_air / Levap

    B0 = grav * (
        wt / (STair + airsea.constants.CtoK) + 0.61 * wq
    )  # surface buoyancy flux
    ustar = coare_u2ustar(
        U10N,
        input_string="u2ustar",
        coare_version="coare3.5",
        TairC=STair,
        z=10,
        zeta=0,
    )
    LMO = -(ustar * ustar * ustar) / vKarman / B0  # Monin Obukove Length scale
    return np.squeeze(LMO)


def PHIu(zeta, option="Hogstroem_1988"):
    """
        Nondimensional stability function PHIu(zeta=z/L) for wind speed

        :param zeta: data series of z/L
        :param option: option to specify which literature functionality to used options = ['Hogstroem_1988']
                
        :returns: PHIu: Nondimensional stability function
    """
    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan] = 0
    option_list = ["Hogstroem_1988"]

    if option == "Hogstroem_1988":
        phi = 1 + 5 * zeta
        phi[zeta < 0] = np.power((1 - 16 * zeta[zeta < 0]), -0.25)
    else:
        print("unexpected option! available options are:")
        print(option_list)
        phi = []
    phi[isnan] = np.nan
    return np.squeeze(phi)


def PHIh(zeta, option="Hogstroem_1988"):
    """
        Nondimensional stability function PHIu(zeta=z/L) for scalars e.g. temperature/humidity

        :param zeta: data series of z/L
        :param option: option to specify which literature functionality to used options = ['Hogstroem_1988']
                
        :returns: PHIh: Nondimensional stability function
    """
    import numpy as np

    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan] = 0
    option_list = ["Hogstroem_1988"]

    if option == "Hogstroem_1988":
        phi = np.power((1 + 4 * zeta[zeta < 0]), 2)
        phi[zeta < 0] = np.power((1 - 16 * zeta[zeta < 0]), -0.5)
    else:
        print("unexpected option! available options are:")
        print(option_list)
        phi = []
    phi[isnan] = np.nan
    return np.squeeze(phi)


def PSIh(zeta, option="Brandt_2002"):
    """
        PSIh(zeta=z/L) is the integral of the ondimensional stability function PHIh(zeta=z/L) for scalars e.g. temperature/humidity

        :param zeta: data series of z/L
        :param option: option to specify which literature functionality to used options = ['Brandt_2002']
                
        :returns: PSIh: Nondimensional stability function
    """
    import numpy as np

    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan] = 0
    option_list = ["Brandt_2002"]

    if option == "Brandt_2002":
        psi = -5 * zeta
        psi[zeta < 0] = np.exp(
            0.598
            + 0.390 * np.log(-zeta[zeta < 0])
            - 0.09 * np.power(np.log(-zeta[zeta < 0]), 2)
        )
    else:
        print("unexpected option! available options are:")
        print(option_list)
        psi = []
    psi[isnan] = np.nan
    return np.squeeze(psi)


def PSIu(zeta, option="Fairall_1996"):
    """
        PSIu(zeta=z/L) is the integral of the ondimensional stability function PHIu(zeta=z/L) for scalars e.g. temperature/humidity

        :param zeta: data series of z/L
        :param option: option to specify which literature functionality to used options = [Dyer_Hicks_1970, 'Fairall_1996']
                
        :returns: PSIh: Nondimensional stability function
    """

    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan] = 0
    if option == "Dyer_Hicks_1970":  # or Dyer_Hicks_1970
        # Dyer and Hicks 1970
        x = zeta * 0  # avoid warings
        x[zeta < 0] = np.sqrt(np.sqrt(1 - 15 * zeta[zeta < 0]))
        # sqrt(sqrt) instead of ^.25
        psi = (
            2 * np.log((1 + x) / 2)
            + np.log((1 + x * x) / 2)
            - 2 * np.arctan(x)
            + 2 * np.arctan(1)
        )
        psi[zeta >= 0] = -5 * zeta[zeta >= 0]
    elif option == "Fairall_1996":
        xk = np.power((1 - 16 * zeta), 0.25)
        xc = np.power((1 - 12.87 * zeta), 0.3333)

        psik = (
            2 * np.log((1 + xk) / 2)
            + np.log((1 + xk * xk) / 2)
            - 2 * np.arctan(xk)
            + 2 * np.arctan(1)
        )

        psic = (
            1.5 * np.log((1 + xc + xc * xc) / 3)
            - np.sqrt(3) * np.arctan((1 + 2 * xc) / np.sqrt(3))
            + 4 * np.arctan(1) / np.sqrt(3)
        )
        f = 1 / (1 + zeta * zeta)
        psi = (1 - f) * psic + f * psik
        c = np.min([50 * np.ones_like(zeta), 0.35 * zeta], axis=0)
        psi[zeta > 0] = -(
            (1 + 1.0 * zeta[zeta > 0])
            + 0.667 * (zeta[zeta > 0] - 14.28) / np.exp(c[zeta > 0])
            + 8.525
        )
    else:
        print('unexpected option: please use "default"')
        psi = []
    psi[isnan] = np.nan
    return np.squeeze(psi)


def coare_u2ustar(
    u, input_string="u2ustar", coare_version="coare3.5", TairC=20.0, z=10.0, zeta=0.0
):
    """
        The function coare_u2ustar uses the wind speed dependend drag coefficient to iteratively convert between u* and uz
        for details and as reference please see https://www.atmos-chem-phys.net/18/4297/2018/ equation (4),(5), and (6)
        
        :param u: input velocity scale either u* or u(z) (the way it is processed depends on input_string).
        :param input_string: either 'u2ustar' or 'ustar2u'
            for input_string=='u2ustar': coare_u2ustar converts u(z)->u*
            for input_string=='ustar2u': coare_u2ustar converts u*->u(z)
        : param coare_version: defines the neutral drag coefficient. Valid input in ['coare3.0', 'coare3.5']
            coare_version='coare3.5' use wind speed dependend charnock coefficient coare version 3.5 Edson et al. 2013
            coare_version='coare3.0' use wind speed dependend charnock coefficient coare version 3.0 Fairall et al. 2003
        : param TairC: air temperature in degree C (has neglegible influence on the results)
        : param z: measurement height in meter
        : param zeta: zeta=z/L nondimensional stability parameter
        
        :returns: u: ouput velocity scale either u* or u(z) depending on the param input_string
        
    """
    import numpy as np

    z0 = 1e-4  # default roughness length (could calculate this using first guess charnock and ustar)

    if type(u) != np.ndarray:
        u = np.asarray([u])
    if type(TairC) != np.ndarray:
        TairC = np.asarray([TairC])
    if type(z) != np.ndarray:
        z = np.asarray([z])
    if type(zeta) != np.ndarray:
        zeta = np.asarray([zeta])


    if input_string == "ustar2u":
        ustar = u
        u10n = 30 * ustar
        # first guess
    elif input_string == "u2ustar":
        u10n = u * np.log(10 / z0) / np.log(z / z0)
        # first guess u10n for calculating initial charnock
        ustar = u10n / 30
    else:
        print('unexpected "input_string"! please use "u2ustar" or "ustar2u"')

    #
    vKarman = airsea.constants.kappa
    # van Karman constant
    grav = airsea.constants.g
    # const of gravitation

    t = TairC
    # air temperature [C]
    gamma = 0.11
    # roughness Reynolds number
    charnock = 0.011
    # first guess charnock parameter (not used)
    visa = 1.326e-5 * (1 + 6.542e-3 * t + 8.301e-6 * t * t - 4.84e-9 * t * t * t)
    # viscosity of air

    for jj in [1, 2, 3, 4, 5, 6]:
        if coare_version == "coare3.5":
            charnock = 0.0017 * u10n - 0.005
            # note EDSON2013 gives this as 0.017*U10-0.005 BUT from plot it must be 0.0017!!!
            charnock[u10n > 19.4] = 0.028
            # charnock(19.4)~0.028
        elif coare_version == "coare3.0":
            charnock = 0.00225 + 0.007 / 8 * u10n
            # Fairall2003 a=0.011@u=10 and a=0.018@u=18
            charnock[u10n > 18] = 0.018
            charnock[u10n < 10] = 0.011
        else:
            print('unexpected "coare_version"! please use "coare3.5" or "coare3.0"')

        # with updated charnock (and ustar) re-calcualte z0 and the Drag Coefficient
        z0 = gamma * (visa / ustar) + charnock * ustar * ustar / grav
        sqrt_C_D = vKarman / np.log(z / z0)
        sqrt_C_D = vKarman / (np.log(z / z0) - PSIu(zeta, option="Fairall_1996"))
        # when adding stability use this equation ...
        sqrt_C_D_10 = vKarman / np.log(10 / z0)
        # 10m neutral drag coefficient

        if input_string == "ustar2u":
            # ustar stays const (input)
            # u and u10n are updated
            u10n = ustar / sqrt_C_D_10
            # update u10n for estimation of charnock
            u = ustar / sqrt_C_D
            # update u
        elif input_string == "u2ustar":
            # u stays const (input)
            # ustar and u10n are updated
            # ustar=(u10n*sqrt_C_D_10);
            ustar = u * sqrt_C_D
            # update ustar
            u10n = (
                u * np.log(10 / z0) / np.log(z / z0)
            )  # update u10n for estimation of charnock
            # the following would be equivalent ...
            # u10n=(ustar/sqrt_C_D_10); #=u*(vkarman/np.log(z/z0))/(vkarman/np.log(10/z0))

    if input_string == "u2ustar":
        u = ustar  # return ustar in this case
        # in the other case (ustar2u) u is already what we want to return

    return np.squeeze(u)


# some sea water properties
def roh_sea(SST, SSS):
    """
        Sea water density at standard atmospheric pressure (1 amt) as function of sea surface temperature and salinity
        Refrence: https://www.tandfonline.com/doi/abs/10.5004/dwt.2010.1079

        :param SST: sea surface temperature in [degree Celsius]
        :param SSS: sea surface salinity in [g/kg]
                
        :returns: rho_sea: Sea water density at standard atmospheric pressure (1 amt) in [kg/m3]
    """

    t = SST
    S = SSS / 1000
    a1 = 9.999 * 1e2
    a2 = 2.034 * 1e-2
    a3 = -6.162 * 1e-3
    a4 = 2.261 * 1e-5
    a5 = -4.657 * 1e-8
    b1 = 8.020 * 1e2
    b2 = -2.001
    b3 = 1.677 * 1e-2
    b4 = -3.060 * 1e-5
    b5 = -1.613 * 1e-5
    rho_sea = (
        a1
        + t * (a2 + t * (a3 + t * (a4 + a5 * t)))
        + b1 * S
        + b2 * S * t
        + b3 * S * t * t
        + b4 * S * t * t * t
        + b5 * S * S * t * t
    )  # (8)
    # Validity: ρsw in (kg/m3); 0 < t < 180 oC; 0 < S < 0.16 kg/kg
    # Accuracy: ±0.1 %
    return rho_sea  # kg/m3


def dynamic_viscosity_sea(SST, SSS):
    """
        Dynamic viscosity of sea water as function of sea surface temperature and salinity
        Refrence: https://www.tandfonline.com/doi/abs/10.5004/dwt.2010.1079

        :param SST: sea surface temperature in [degree Celsius]
        :param SSS: sea surface salinity in [g/kg]
                
        :returns: musw: Dynamic viscosity of sea water in [kg/m/s]
    """
    t = SST
    S = SSS / 1000  # g/kg -> kg/kg
    # @ 5C @ 35PSU

    # https://www.tandfonline.com/doi/abs/10.5004/dwt.2010.1079
    # dynamic viscosity
    # μw is based on the IAPWS 2008 [73] data and given by
    muw = 4.2844 * 1e-5 + 1 / (0.157 * (t + 64.993) * (t + 64.993) - 91.296)  # eq. (23)

    A = 1.541 + 1.998 * 1e-2 * t - 9.52 * 1e-5 * t * t
    B = 7.974 - 7.561 * 1e-2 * t + 4.724 * 1e-4 * t * t

    musw = muw * (1 + S * (A + B * S))  # (22)

    # Validity: μsw and μw in (kg/m.s); 0 < t < 180 oC; 0 < S < 0.15 kg/kg
    # Accuracy: ±1.5 %

    return musw  # [kg/m/s]


def kinematic_viscosity_sea(SST, SSS):
    """
        Kinematic viscosity of sea water as function of sea surface temperature and salinity
        Refrence: https://www.tandfonline.com/doi/abs/10.5004/dwt.2010.1079

        :param SST: sea surface temperature in [degree Celsius]
        :param SSS: sea surface salinity in [g/kg]
                
        :returns: nusw: kinematic viscosity of sea water in [m2/s]
    """

    roh_sw = roh_sea(SST, SSS)
    musw = dynamic_viscosity_sea(SST, SSS)
    nusw = musw / roh_sw
    return nusw  # kinematic viscosity in [m2/s]


def wet_bulb_temperature(TA, RH):
    """
        Wet bulb temperature as function of air temperature and relative humidty
        Refrence: https://journals.ametsoc.org/doi/full/10.1175/JAMC-D-11-0143.1: Roland Stull "Wet-Bulb Temperature from Relative Humidity and Air Temperature"

        :param TA: air temparature [degree Celsius]
        :param RH: relative humidity [%]
                
        :returns: TW: wet bulb temperature [degree Celsius]
    """
    # https://journals.ametsoc.org/doi/full/10.1175/JAMC-D-11-0143.1
    # Roland Stull "Wet-Bulb Temperature from Relative Humidity and Air Temperature"

    # Tw = T atan[0.151977(RH% + 8.313659)^1/2] + atan(T + RH%) - atan(RH% - 1.676331) + 0.00391838(RH%)^3/2*atan(0.023101RH%) - 4.686035
    TW = (
        TA * np.arctan(0.151977 * np.power(RH + 8.313659, 0.5))
        + np.arctan(TA + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * np.power(RH, 1.5) * np.arctan(0.023101 * RH)
        - 4.686035
    )
    return TW


def water_vapour_saturation_pressure(TA, PA, SSS=35):
    """
        Vapor Pressure and Enhancement as function of air temperature and atmospheric pressure and sea water salinity
        Refrence: Arden L. Buck, New Equations for Computing Vapor Pressure and Enhancement Factor, Journal of Applied Meterology, December 1981, Volume 20, Page 1529.

        :param TA: air temparature [degree Celsius]
        :param PA: atmospheric pressure [hPa]
        :param SSS: sea surface salinity in [PSU]
                
        :returns: e_sat: saturation pressure of water vapour [hPa]
    """
    e_sat = (6.1121) * (1.0007 + 3.46e-6 * PA) * np.exp(17.502 * TA / (240.97 + TA))
    e_sat = e_sat * (1 - 0.000537 * SSS)
    return e_sat
