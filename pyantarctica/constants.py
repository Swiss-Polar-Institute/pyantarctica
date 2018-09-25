# list of constants

# physical constants

# van Karman constant used in Boundary layer theory most accepted value = 0.4 (dimensionless)
vanKarman = 0.4

#Cp: specific heat capacity of (dry) air at constant pressure 
# at 300K "Tables of Thermal Properties of Gases", NBS Circular 564,1955.
# https://www.ohio.edu/mechanical/thermo/property_tables/air/air_cp_cv.html
#Cp = constants.Cp
Cp = 1005 # J/kg/K

# 
Levap = 2.5e+6 # J/kg

rho_air = 1 # kg/m3

# gravity [m/s^2]
g = 9.81
#def Gravity():
    # could define gravity based on latitude
#    g=9.81 # const of gravitation
#    return g
    #An alternative formula for g as a function of latitude is the WGS (World Geodetic System) 84 Ellipsoidal Gravity Formula:[17]
    #g { ϕ } = G e [ 1 + k sin 2 ⁡ ϕ 1 − e 2 sin 2 ⁡ ϕ ] , {\displaystyle g\{\phi \}=\mathbb {G} _{e}\left[{\frac {1+k\sin ^{2}\phi }{\sqrt {1-e^{2}\sin ^{2}\phi }}}\right],\,\!} g\{\phi\}= \mathbb{G}_e\left[\frac{1+k\sin^2\phi}{\sqrt{1-e^2\sin^2\phi}}\right],\,\!