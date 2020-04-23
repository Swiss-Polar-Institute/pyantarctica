# some usefull functions based on air-sea literature
# I have coded these to my best knowledge, but I cannot quarantee for the content to be correct.
# last reviewed by Sebastian Landwehr PSI 27.03.2020
# uses the airsea toolbox of Filipe Fernandes
# - # pip install airsea
# - # https://github.com/pyoceans/python-airsea
# - # https://pypi.org/project/airsea/

import numpy as np

def LMoninObukov_bulk(U10,SSHF,SLHF,STair):
    # Monin Obukov Length scale as function of
    # U10 = 10 meter neutral wind speed
    # SSHF = surface sensible heat flux [W/m2]
    # SLHF = surface latent heat flux [W/m2]
    # STair = surface air temperature [C]
    
    import airsea # 

    if type(U10) != np.ndarray:
        U10 = np.array([U10])
    if type(SSHF) != np.ndarray:
        SSHF = np.array([SSHF])
    if type(SLHF) != np.ndarray:
        SLHF = np.array([SLHF])
    if type(STair) != np.ndarray:
        STair = np.array([STair])
        
    Cp = airsea.constants.cp # 1004.7 or 1005 # J/kg/K
    Levap = airsea.atmosphere.vapor(STair) # ~2.5e+6 J/kg
    rho_air = airsea.atmosphere.air_dens(Ta=STair, rh=(STair*0))
    vKarman = airsea.constants.kappa; # van Karman constant
    grav = airsea.constants.g; # const of gravitation
    
    wt = SSHF/rho_air/Cp
    wq = SLHF/rho_air/Levap
    
    B0 = grav*(wt/(STair+airsea.constants.CtoK) + 0.61*wq) # surface buoyancy flux
    ustar = coare_u2ustar (U10, input_string='u2ustar', coare_version='coare3.5', TairC=STair, z=10, zeta=0)
    LMO = -(ustar*ustar*ustar)/vKarman/B0 # Monin Obukove Length scale
    return np.squeeze(LMO)

def PHIu(zeta, option='Hogstroem_1988'):
    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan]=0
    option_list = ['Hogstroem_1988']
    
    if option == 'Hogstroem_1988':
        phi = 1+5*zeta
        phi[zeta<0]=np.power((1-16*zeta[zeta<0]),-0.25)
    else:
        print('unexpected option! available options are:')
        print(option_list)
        phi = []
    phi[isnan]=np.nan   
    return np.squeeze(phi)

def PHIh(zeta, option='Hogstroem_1988'):
    import numpy as np
    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan]=0
    option_list = ['Hogstroem_1988']
    
    if option == 'Hogstroem_1988':
        phi = np.power((1+4*zeta[zeta<0]),2)
        phi[zeta<0]=np.power((1-16*zeta[zeta<0]),-0.5)
    else:
        print('unexpected option! available options are:')
        print(option_list)
        phi = []
    phi[isnan]=np.nan   
    return np.squeeze(phi)

def PSIh(zeta, option='Brandt_2002'):
    import numpy as np
    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan]=0
    option_list = ['Brandt_2002']
    
    if option == 'Brandt_2002':
        psi=-5*zeta
        psi[zeta<0] = np.exp(0.598+0.390*np.log(-zeta[zeta<0])-0.09*np.power(np.log(-zeta[zeta<0]),2) )
    else:
        print('unexpected option! available options are:')
        print(option_list)
        psi = []
    psi[isnan]=np.nan   
    return np.squeeze(psi)

def PSIu(zeta, option='Fairall_1996'):
    #stability correction function for modifying the logarithmic wind speed profiles based on atmospheric stability
    # use e.g. for: u(z)=u*/k[log(z/z0)-PSIu(z/L)]
    #
    # PSIu is integral of the semiempirical function PHIu
    # PSIu(z/L)=INT_z0^z[1-PHI_u(z/L)]d(z/L)/(z/L)
    # several forms of PHIu and PSIu are published and will be added as options
    # default = 'Dyer_Hicks_1970'
    
    import numpy as np
    # zeta=z/L 
    # with L = -u*^3/vkarman/(g<wT>/T+0.61g<wq>)
    #x=np.sqrt(np.sqrt(1-15*zeta)); #sqrt(sqrt) instead of ^.25
    
    zeta = np.asarray([zeta])
    isnan = np.isnan(zeta)
    zeta[isnan]=0
    if option == 'Dyer_Hicks_1970': # or Dyer_Hicks_1970
        # Dyer and Hicks 1970       
        x=zeta*0 # avoid warings
        x[zeta<0]=np.sqrt(np.sqrt(1-15*zeta[zeta<0])); #sqrt(sqrt) instead of ^.25
        psi=2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x)+2*np.arctan(1); 
        psi[zeta>=0]=-5*zeta[zeta>=0];
    elif option == 'Fairall_1996':
        xk = np.power( (1-16*zeta) , .25)
        xc = np.power( (1-12.87*zeta) , .3333)

        psik=2*np.log((1+xk)/2)+np.log((1+xk*xk)/2)-2*np.arctan(xk)+2*np.arctan(1); 

        psic=1.5*np.log((1+xc+xc*xc)/3)-np.sqrt(3)*np.arctan((1+2*xc)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3);
        f=1/(1+zeta*zeta);
        psi=(1-f)*psic+f*psik;
        c=np.min([50*np.ones_like(zeta),.35*zeta],axis=0);
        psi[zeta>0]=-((1+1.0*zeta[zeta>0]) +.667*(zeta[zeta>0]-14.28)/np.exp(c[zeta>0])+8.525);
    else:
        print('unexpected option: please use "default"')
        psi = []
    psi[isnan]=np.nan   
    return np.squeeze(psi)


def coare_u2ustar (u, input_string='u2ustar', coare_version='coare3.5', TairC=20.0, z=10.0, zeta=0.0): 
    # function coare_u2ustar (u,coare_direction,coare_version) 
    # uses wind speed dependend drag coefficient to iteratively convert between u* and uz
    #
    # the input is procesed dependend on the input_string
    # for input_string=='u2ustar': coare_u2ustar converts u(z)(neutral conditions assumed)->u*
    # for input_string=='ustar2u': coare_u2ustar converts u*->u(z)(neutral conditions assumed)
    #
    # coare_version defines which drag coefficient is used for the conversion
    # coare_version='coare3.5' use wind speed dependend charnock coefficient coare version 3.5 Edson et al. 2013
    # coare_version='coare3.0' use wind speed dependend charnock coefficient coare version 3.0 Fairall et al. 2003
    # for citing this code please refere to:  
    # https://www.atmos-chem-phys.net/18/4297/2018/ equation (4),(5), and (6)
    # Sebastian Landwehr, PSI 2018
    import numpy as np
    import airsea # airsea toolbox of Filipe Fernandes [don't mix up with this one!]

    z0 = 1e-4 # default roughness length (could calculate this using first guess charnock and ustar)
    
    if type(u) != np.ndarray:
        u = np.asarray([u])
    if type(TairC) != np.ndarray:
        TairC = np.asarray([TairC])
    if type(z) != np.ndarray:
        z = np.asarray([z])
    if type(zeta) != np.ndarray:
        zeta = np.asarray([zeta])


    import numpy as np
    if input_string == 'ustar2u':
        ustar = u;
        u10n = 30*ustar; # first guess
    elif input_string == 'u2ustar':
        u10n = u*np.log(10/z0)/np.log(z/z0); # first guess u10n for calculating initial charnock
        ustar = u10n/30;
    else:
        print('unexpected "input_string"! please use "u2ustar" or "ustar2u"')
        
    #
    vKarman = airsea.constants.kappa; # van Karman constant
    grav = airsea.constants.g; # const of gravitation
    
    t=TairC; # air temperature [C]
    gamma = 0.11; # roughness Reynolds number
    charnock = 0.011; # first guess charnock parameter (not used)
    visa=1.326e-5*(1+6.542e-3*t+8.301e-6*t*t-4.84e-9*t*t*t); # viscosity of air



    for jj in [1, 2, 3, 4, 5, 6]:
        if coare_version == 'coare3.5':
            charnock=0.0017*u10n-0.005; # note EDSON2013 gives this as 0.017*U10-0.005 BUT from plot it must be 0.0017!!!
            charnock[u10n>19.4]=0.028; # charnock(19.4)~0.028
        elif coare_version == 'coare3.0':
            charnock=0.00225+0.007/8*u10n; # Fairall2003 a=0.011@u=10 and a=0.018@u=18
            charnock[u10n>18]=0.018; 
            charnock[u10n<10]=0.011; 
        else:
            print('unexpected "coare_version"! please use "coare3.5" or "coare3.0"')

        # with updated charnock (and ustar) re-calcualte z0 and the Drag Coefficient
        z0 = gamma*(visa/ustar)+charnock*ustar*ustar/grav;
        sqrt_C_D = (vKarman/np.log(z/z0));
        sqrt_C_D = (vKarman/(np.log(z/z0)-PSIu(zeta, option = 'Fairall_1996'))); # when adding stability use this equation ...
        sqrt_C_D_10 = (vKarman/np.log(10/z0)); # 10m neutral drag coefficient

        if input_string == 'ustar2u':
            #ustar stays const (input)
            #u and u10n are updated
            u10n=(ustar/sqrt_C_D_10); # update u10n for estimation of charnock
            u=(ustar/sqrt_C_D); # update u
        elif input_string == 'u2ustar':
            #u stays const (input)
            #ustar and u10n are updated
            #ustar=(u10n*sqrt_C_D_10);
            ustar=(u*sqrt_C_D); # update ustar
            u10n=u*np.log(10/z0)/np.log(z/z0) # update u10n for estimation of charnock
            # the following would be equivalent ...
            #u10n=(ustar/sqrt_C_D_10); #=u*(vkarman/np.log(z/z0))/(vkarman/np.log(10/z0))
            
    if input_string == 'u2ustar':
        u=ustar # return ustar in this case
        # in the other case (ustar2u) u is already what we want to return
        
    return np.squeeze(u)

# some sea water properties
def roh_sea(SST,SSS):
    # SST in C!
    # SSS in g/kg
    # https://www.tandfonline.com/doi/abs/10.5004/dwt.2010.1079
    # sea water density at atm pressure
    t=SST
    S=SSS/1000
    a1 = 9.999*1E2
    a2 = 2.034*1E-2
    a3 = -6.162*1E-3
    a4 = 2.261*1E-5
    a5 = -4.657*1E-8
    b1 = 8.020*1E2
    b2 = -2.001
    b3 = 1.677*1E-2
    b4 = -3.060*1E-5
    b5 = -1.613*1E-5
    rho_sea = a1 + t*(a2 + t*(a3 + t*(a4 + a5*t))) + b1*S + b2*S*t + b3*S*t*t + b4*S*t*t*t + b5*S*S*t*t #(8)
    #Validity: ρsw in (kg/m3); 0 < t < 180 oC; 0 < S < 0.16 kg/kg
    #Accuracy: ±0.1 %
    return rho_sea # kg/m3

def dynamic_viscosity_sea(SST,SSS):
    t=SST
    S=SSS/1000 # g/kg -> kg/kg
    #@ 5C @ 35PSU

    # https://www.tandfonline.com/doi/abs/10.5004/dwt.2010.1079
    # dynamic viscosity 
    # μw is based on the IAPWS 2008 [73] data and given by
    muw = 4.2844*1E-5 + 1/(0.157*(t+64.993)*(t+64.993)-91.296) # eq. (23)

    A = 1.541 + 1.998*1E-2*t - 9.52*1E-5*t*t
    B = 7.974 - 7.561*1E-2*t+ 4.724*1E-4*t*t

    musw = muw*(1 + S*(A + B*S) ) # (22)

    #Validity: μsw and μw in (kg/m.s); 0 < t < 180 oC; 0 < S < 0.15 kg/kg
    #Accuracy: ±1.5 %

    return musw # [kg/m/s]

def kinematic_viscosity_sea(SST,SSS):
    roh_sw = roh_sea(SST,SSS)
    musw = dynamic_viscosity_sea(SST,SSS)
    nusw = musw/roh_sw
    return nusw # kinematic viscosity in [m2/s]


def wet_bulb_temperature(TA,RH):
    # https://journals.ametsoc.org/doi/full/10.1175/JAMC-D-11-0143.1
    # Roland Stull "Wet-Bulb Temperature from Relative Humidity and Air Temperature"

    #Tw = T atan[0.151977(RH% + 8.313659)^1/2] + atan(T + RH%) - atan(RH% - 1.676331) + 0.00391838(RH%)^3/2*atan(0.023101RH%) - 4.686035
    TW = TA*np.arctan(0.151977*np.power(RH + 8.313659,0.5)) + np.arctan(TA + RH) - np.arctan(RH - 1.676331) + 0.00391838*np.power(RH,1.5)*np.arctan(0.023101*RH) - 4.686035
    return TW

def water_vapour_saturation_pressure(TA,PA,SSS=35):
    # TA Celsius,
    # PA hPa (1hecto Pa = 1 mbar = 0.1*1kilo Pa)
    # SSS PSU
    # returns e_sat in hPa
    e_sat = (6.1121)*(1.0007 + 3.46E-6*PA)*np.exp( 17.502*TA/(240.97+TA))
    # Arden L. Buck, New Equations for Computing Vapor Pressure and Enhancement
    # Factor, Journal of Applied Meterology, December 1981, Volume 20, Page 1529.
    e_sat = e_sat*(1 - 0.000537*SSS)
    return e_sat
