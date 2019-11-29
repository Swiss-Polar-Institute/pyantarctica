# some functions needed in the sea spray source function project
import numpy as np
import pandas as pd
from pathlib import Path
import pyantarctica.aceairsea as aceairsea
import pyantarctica.dataset as dataset
import pyantarctica.datafilter as datafilter


def r_div_r80(RH, option='Zieger2016_'):
    # hygroscopic growth factor
    # (below RH=75% this is valid only for decreasing RH)!
    # THIS NEGLECTS THE KELVIN EFFECT!
    # But the difference should be really small

    RH=RH.copy(); RH[RH>99]=99; #
    if option == 'Lewis2004':
        # within 2.5% of Tang 1997 for RH>50%,
        r_div_r80 = 0.54*np.power((1+1/(1-RH/100)), 1/3)
        #r_div_r80[r_div_r80>2]=2 # limit to 98%?-> gf=2.
        #r_div_r80[RH>98]=2 # limit to 98%?-> gf=2.
        r_div_r80[RH<42]=0.5 # for low RH set rRH=rDry=0.5*r80
    if option == 'Zieger2016_':
        r_div_r80 = (1.201*np.power((1-RH/100), -0.2397))*0.5 # per email
        #r_div_r80[r_div_r80>2]=2 # limit to 98%?-> gf=2.
        #r_div_r80[RH>98]=2 # limit to 98%?-> gf=2.
        r_div_r80[RH<50]=( (1-5*.12/45)+0.12/45*RH[RH<50])*0.5
        r_div_r80[RH<5]=1*0.5


    return r_div_r80

def rdry2rRH(Dp, RH, hygroscopic_growth='Zieger2016_'):
    # convert try diameter/radius to expected diameter/radius at RH[%]
    # rRH/rDry = (rRH/r80)*(r80/rDry) = (rRH/r80)*2
    if type(RH) != np.ndarray:
        RH = np.asarray([RH])
    RH = RH.reshape(len(RH),1)
    if type(Dp) != np.ndarray:
        Dp = np.asarray([Dp])

    if ((len(RH)==len(Dp)) & (len(Dp)>1) ):
        # in this case assume each Dp sample corresponds to an RH sample, output will be (n,)
        Dp = Dp.reshape(len(Dp),1) #
    else:
        # output will be (m,) or (n,m)
        Dp = Dp.reshape(1,len(Dp)) #

    DpRH = 2*r_div_r80(RH, option=hygroscopic_growth)*Dp

    if np.max(np.shape(DpRH))>1:
        DpRH = DpRH.squeeze()
    else:
        DpRH = np.array([DpRH[0]])
        if len(np.shape(DpRH))==2:
            DpRH = DpRH[0]

    return DpRH

def rho_sea_spary(Dp_dry, RH, rho_p=2.2, hygroscopic_growth='Zieger2016_'):
    if type(RH) != np.ndarray:
        RH = np.asarray([RH])
    RH = RH.reshape(len(RH),1)

    if type(Dp_dry) != np.ndarray:
        Dp_dry = np.asarray([Dp_dry])

    if ((len(RH)==len(Dp_dry)) & (len(Dp_dry)>1) ):
        # in this case assume each Dp sample corresponds to an RH sample, output will be (n,)
        Dp_dry = Dp_dry.reshape(len(Dp_dry),1) #
    else:
        # output will be (m,) or (n,m)
        Dp_dry = Dp_dry.reshape(1,len(Dp_dry)) #

    Dp_RH = 2.*r_div_r80(RH, option=hygroscopic_growth)*Dp_dry
    V_dry = 3/4*np.pi*np.power(Dp_dry/2,3)
    V_H2O = 3/4*np.pi*np.power(Dp_RH/2,3)-V_dry
    rho_ss = (1.*V_H2O+rho_p*V_dry)/(V_H2O+V_dry) # new density equals volumn weighted average of the densities

    # ensuring the right shape what ever the input
    if np.max(np.shape(rho_ss))>1:
        rho_ss = rho_ss.squeeze()
    else:
        rho_ss = np.array([rho_ss[0]])
        if len(np.shape(rho_ss))==2:
            rho_ss = rho_ss[0]

    return rho_ss

def sea_salt_settling_velocity(Dp, rho_p=2.2, RH=80., T=20., P=1013., hygroscopic_growth='Zieger2016_'):

    # Required input:
    # Dp = aerosol diameter [um] as (n,) numpy.array with n>=1
    # rho_p = aerosol density g cm^-3
    # T [C]
    # P [hPa]
    #
    # Output
    # vd = settling velocity [m/s ] per size bin as numpy.array (m,n) $ COMES OUT IN mm/s ???
    # or (n,) if m=1 & n>1
    # or (m,) if n=1 & m>1
    # or (1,) if n=m=1


    T = T+273.15 # C-> K
    P = P*100 # hPa -> Pa

    if type(RH) != np.ndarray:
        RH = np.asarray([RH])
    if type(P) != np.ndarray:
        P = np.asarray([P])
    if type(T) != np.ndarray:
        T = np.asarray([T])


    T = T.reshape(len(T),1)
    P = P.reshape(len(P),1)
    RH = RH.reshape(len(RH),1)

    # convert denisty and diameter based on Relative Humidity [%]
    # THE rho_sea_spary FUNCTION REQUIRES rho_p to be in g cm^-3 !!!!!!!!!!!!!!!!!!!!!!!!!!!
    rho_p = rho_sea_spary(Dp, RH, rho_p=rho_p, hygroscopic_growth=hygroscopic_growth)
    Dp = rdry2rRH(Dp, RH, hygroscopic_growth=hygroscopic_growth)

    
    rho_p = rho_p * 100*100*100/1000 # g cm^-3 -> kg m^-3
    Dp = Dp*1E-6 # um -> m

    
    Ccunn = 1 # need to parametrise base ond Dp, RH!
    dyn_visc = 0.000018 #kg m−1 s−1 dynamic viscosity of air
    g = 9.81 # kg m−1 s−2
    R = 8.314 # Nm/mol/K
    M = 28.9647/1000 # kg/mol
    kBolz = 1.38*1E-23 # m2 kg s-2 K-1

    kin_visc = 1.5E-5 # m^2/s  kinematic viscosity of air (depends on temperature!) ?
    # mean free path of air molecules
    if 0:
        mfp = 2*dyn_visc/(P*np.sqrt(*M/(np.pi*R*T ))) # ~0.0651 um
        # varying p-> *0.7, T-> -40K changes mfp by only 30%
        # change in Ccunn 6%
        # alsomost invisible in vg
    else:
        mfp = 6.511*1E-8
    Ccunn = 1+mfp/Dp*(2.514+0.8*np.exp(-0.55*Dp/mfp)) # Seinfeld Pandis 8.34
    # Ccunn varies from 1.2 for Dp=.8um to 1.032 for Dp=5um

    # setttling velocity in m/s,note that the Dp and rho_p play an important role!
    vs = g*rho_p*Dp*Dp*Ccunn/18/dyn_visc


    if np.max(np.shape(vs))>1:
        vs = vs.squeeze()
    else:
        vs = np.array([vs[0]])

    return vs

def sea_salt_deposition_velocity(Dp_dry, rho_dry=2.017, h_ref=15., U10=10., RH=80., T=20., P=1013., zeta=0., model='giardina_2018', hygroscopic_growth='Zieger2016_'):

#if 1:
    # Required input:
    # Dp = aerosol diameter [um] as (n,) numpy.array with n>=1
    # rho_p = aerosol density g cm^-3!!
    # h_ref = 15 [m] reference height
    # U10 = wind speed [m/s] referenced to 10m, neutral stability as (m,) numpy.array with m>=1
    # T [C]
    # P [hPa]
    #
    # Output
    # vd = deposition velocity [m/s ] per size bin as numpy.array (m,n)
    # or (n,) if m=1 & n>1
    # or (m,) if n=1 & m>1
    # or (1,) if n=m=1
    # vs = settiling velocity in the same shape as vd
    import pyantarctica.aceairsea as aceairsea
    import numpy as np

    #zeta=0.; T=20.; P=1013.; h_ref=15.;
    #T=params['TA'].values
    #hygroscopic_growth='Zieger2016_'
    #U10 = params['u10'].values
    #U10 = 1.
    #Dp_dry = APS.columns.astype('float').values # um
    #rho_dry = 2.017
    #RH = params[RH_str].values
    
    if type(zeta) != np.ndarray:
        zeta = np.asarray([zeta])
    if type(U10) != np.ndarray:
        U10 = np.asarray([U10])
    if type(P) != np.ndarray:
        P = np.asarray([P])
    if type(T) != np.ndarray:
        T = np.asarray([T])

    U10 = U10.reshape(len(U10),1)
    T = T.reshape(len(T),1)
    P = P.reshape(len(P),1)
    zeta = zeta.reshape(len(zeta),1)

    # setttling velocity in m/s,note that the Dp and rho_p play an important role!
    #vs = g*rho_p*Dp*Dp*Ccunn/18/dyn_visc
    vs = sea_salt_settling_velocity(Dp_dry, rho_p=rho_dry, RH=RH, T=20., P=1013., hygroscopic_growth=hygroscopic_growth)
    rho_p = rho_sea_spary(Dp_dry, RH=RH, rho_p=rho_dry, hygroscopic_growth=hygroscopic_growth) # rho_p [g cm^-3]
    Dp = rdry2rRH(Dp_dry, RH=RH, hygroscopic_growth=hygroscopic_growth) # Dp [um]
    
    # convert to SI units
    rho_p = rho_p * 100*100*100/1000 # g cm^-3 -> kg m^-3
    Dp = Dp*1E-6 # um -> m

    # friction velocity [m/s]
    ustar = aceairsea.coare_u2ustar(U10, input_string='u2ustar', coare_version='coare3.5', TairC=T, z=10, zeta=0)
    ustar =   np.array([ustar])
    ustar = ustar.reshape(np.max(np.shape(ustar)),1)
    
    T = T+273.15 # C-> K
    P = P*100 # hPa -> Pa
    
    Ccunn = 1 # need to parametrise base ond Dp, RH!
    dyn_visc = 0.000018 #kg m−1 s−1 dynamic viscosity of air
    g = 9.81 # kg m−1 s−2
    R = 8.314 # Nm/mol/K
    M = 28.9647/1000 # kg/mol
    kBolz = 1.38*1E-23 # m2 kg s-2 K-1

    kin_visc = 1.5E-5 # m^2/s  kinematic viscosity of air (depends on temperature!) ?
    # mean free path of air molecules
    if 0:
        mfp = 2*dyn_visc/(P*np.sqrt(8*M/(np.pi*R*T ))) # ~0.0651 um
        # varying p-> *0.7, T-> -40K changes mfp by only 30%
        # change in Ccunn 6%
        # alsomost invisible in vg
    else:
        mfp = 6.511*1E-8
    Ccunn = 1+mfp/Dp*(2.514+0.8*np.exp(-0.55*Dp/mfp)) # Seinfeld Pandis 8.34
    # Ccunn varies from 1.2 for Dp=.8um to 1.032 for Dp=5um

    # Diffusivity
    Diffusivity = kBolz*T*Ccunn/3/np.pi/dyn_visc/Dp # Diffusivity of the aerosol in air
    
    kappa = 0.4
    Pr = 0.72 # Prandtl number
    Sc = kin_visc/Diffusivity # Schmidt number
    z0=0.016*ustar*ustar/g # roughness length ! Check
    ra = 1/kappa/ustar*(np.log(h_ref/z0) - aceairsea.PSIh(zeta) ) # -Psi(z/L)+Pis(z0/L) ! Need to add
    rb = 2/kappa/ustar*np.power(Sc/Pr,2/3) #
    vd = vs + 1/(ra+rb+ra*rb) # addition currenlty makes almost no difference for Dp>.8

    # Giradina 2018
    m=0.1; n=3; # tunig params
    St = vs/g*ustar*ustar/kin_visc # (eq. 23)
    tau = rho_p*Dp*Dp*Ccunn/18/dyn_visc
    tau_plus = tau*ustar*ustar/kin_visc
    rdb = 1/ustar*np.power(Sc,2/3) # (eq. 20)
    rii = 1/ustar/(St*St/(St*St+1)) # eq. 22 for rough surfaces
    rii = 1/ustar/(St*St/(St*St+400)) # eq. 22 for smooth surfaces

    rti = 1/ustar/m/tau_plus
    vd = vs/(1-np.exp(-vs*(ra+1/(1/rdb+1/rii+1/(rii+rti)  )  )) )

    if 0:
        # ensuring the right shape what ever the input
        if np.max(np.shape(vd))>1:
            vd = vd.squeeze()
        else:
            vd = np.array([vd[0]])
            if len(np.shape(vd))==2:
                vd = vd[0]

        if np.max(np.shape(vs))>1:
            vs = vs.squeeze()
        else:
            vs = np.array([vs[0]])

    return vd

def deposition_velocity(Dp, rho_p=2.2, h_ref=15., U10=10., T=20., P=1013., zeta=0., model='giardina_2018'):

#if 1:
    # Required input:
    # Dp = aerosol diameter [um] as (n,) numpy.array with n>=1
    # rho_p = aerosol density g cm^-3
    # h_ref = 15 [m] reference height
    # U10 = wind speed [m/s] referenced to 10m, neutral stability as (m,) numpy.array with m>=1
    # T [C]
    # P [hPa]
    #
    # Output
    # vd = deposition velocity [m/s ] per size bin as numpy.array (m,n)
    # or (n,) if m=1 & n>1
    # or (m,) if n=1 & m>1
    # or (1,) if n=m=1
    # vs = settiling velocity in the same shape as vd
    import pyantarctica.aceairsea as aceairsea
    import numpy as np

    if type(zeta) != np.ndarray:
        zeta = np.asarray([zeta])
    if type(U10) != np.ndarray:
        U10 = np.asarray([U10])
    if type(P) != np.ndarray:
        P = np.asarray([P])
    if type(T) != np.ndarray:
        T = np.asarray([T])
    if type(rho_p) != np.ndarray:
        rho_p = np.asarray([rho_p])

    U10 = U10.reshape(len(U10),1)
    T = T.reshape(len(T),1)
    P = P.reshape(len(P),1)
    zeta = zeta.reshape(len(zeta),1)

    if type(Dp) != np.ndarray:
        Dp = np.asarray([Dp])

    if ((len(U10)==len(Dp)) & (len(Dp)>1) ):
        # in this case assume each Dp sample corresponds to a U10 sample and we get a time series of vd
        Dp = Dp.reshape(len(Dp),1) #
    else:
        Dp = Dp.reshape(1,len(Dp)) #
    #if ((len(U10)==len(rho_p)) & (len(rho_p)>1) ):
    #    # in this case assume each Dp sample corresponds to a U10 sample and we get a time series of vd
    #    rho_p = rho_p.reshape(len(rho_p),1) #

    if len(np.shape(rho_p))==2:
        rho_p = rho_p.reshape(len(U10),len(Dp))
    elif len(np.shape(rho_p))==1:
        if ((len(U10)==len(rho_p)) & (len(rho_p)>1) ):
            rho_p = rho_p.reshape(len(rho_p),1)
        elif ((len(Dp)==len(rho_p)) & (len(rho_p)>1) ):
            rho_p = rho_p.reshape(1,len(rho_p))



    ustar = aceairsea.coare_u2ustar(U10, input_string='u2ustar', coare_version='coare3.5', TairC=T, z=10, zeta=0)
    ustar = aceairsea.coare_u2ustar(U10, input_string='u2ustar', coare_version='coare3.5', TairC=T, z=10, zeta=0)
    ustar =   np.array([ustar])
    ustar = ustar.reshape(np.max(np.shape(ustar)),1)
    #print(ustar)
    rho_p = rho_p * 100*100*100/1000 # g cm^-3 -> kg m^-3
    Dp = Dp*1E-6 # um -> m

    T = T+273.15 # C-> K
    P = P*100 # hPa -> Pa



    #ustar = p.reshape(len(ustar),1)

    Ccunn = 1 # need to parametrise base ond Dp, RH!
    dyn_visc = 0.000018 #kg m−1 s−1 dynamic viscosity of air
    g = 9.81 # kg m−1 s−2
    R = 8.314 # Nm/mol/K
    M = 28.9647/1000 # kg/mol
    kBolz = 1.38*1E-23 # m2 kg s-2 K-1

    kin_visc = 1.5E-5 # m^2/s  kinematic viscosity of air (depends on temperature!) ?
    # mean free path of air molecules
    if 0:
        mfp = 2*dyn_visc/(P*np.sqrt(8*M/(np.pi*R*T ))) # ~0.0651 um
        # varying p-> *0.7, T-> -40K changes mfp by only 30%
        # change in Ccunn 6%
        # alsomost invisible in vg
    else:
        mfp = 6.511*1E-8
    Ccunn = 1+mfp/Dp*(2.514+0.8*np.exp(-0.55*Dp/mfp)) # Seinfeld Pandis 8.34
    # Ccunn varies from 1.2 for Dp=.8um to 1.032 for Dp=5um


    # setttling velocity in m/s,note that the Dp and rho_p play an important role!
    vs = g*rho_p*Dp*Dp*Ccunn/18/dyn_visc

    # Diffusivity
    Diffusivity = kBolz*T*Ccunn/3/np.pi/dyn_visc/Dp # Diffusivity of the aerosol in air


    kappa = 0.4
    Pr = 0.72 # Prandtl number
    Sc = kin_visc/Diffusivity # Schmidt number
    z0=0.016*ustar*ustar/g # roughness length ! Check
    ra = 1/kappa/ustar*(np.log(h_ref/z0) - aceairsea.PSIh(zeta) ) # -Psi(z/L)+Pis(z0/L) ! Need to add
    rb = 2/kappa/ustar*np.power(Sc/Pr,2/3) #


    vs =   np.array([vs])

    vd = vs + 1/(ra+rb+ra*rb) # addition currenlty makes almost no difference for Dp>.8

    # Giradina 2018
    m=0.1; n=3; # tunig params
    St = vs/g*ustar*ustar/kin_visc # (eq. 23)
    tau = rho_p*Dp*Dp*Ccunn/18/dyn_visc
    tau_plus = tau*ustar*ustar/kin_visc
    rdb = 1/ustar*np.power(Sc,2/3) # (eq. 20)
    rii = 1/ustar/(St*St/(St*St+1)) # eq. 22 for rough surfaces
    rii = 1/ustar/(St*St/(St*St+400)) # eq. 22 for smooth surfaces

    rti = 1/ustar/m/tau_plus
    vd = vs/(1-np.exp(-vs*(ra+1/(1/rdb+1/rii+1/(rii+rti)  )  )) )


    # ensuring the right shape what ever the input
    if np.max(np.shape(vd))>1:
        vd = vd.squeeze()
    else:
        vd = np.array([vd[0]])
        if len(np.shape(vd))==2:
            vd = vd[0]

    if np.max(np.shape(vs))>1:
        vs = vs.squeeze()
    else:
        vs = np.array([vs[0]])

    return vd, vs


def dFdlogD_to_dFdD(dFdlogr80,r80):
    dFdr80 = dFdlogr80/r80*np.log10(np.exp(1))
    return dFdr80

def dFdD_to_dFdlogD(dFdr80,r80):
    dFdlogr80 = dFdr80*r80/np.log10(np.exp(1))
    return dFdlogr80



def sssf(sssf_str, r80, U10, SST=[], Re=[]):
    # sssf(sssf_str, r80, U10, SST, Re):
    #
    # Required input:
    # sssf_str = string denoting the sssf parametrisation to be used
    # r80 = aerosol diameter [um] at 80%RH (r80=2dry=0.5r_formation, based on L&S 2004) as (n,) numpy.array with n>=2
    # U10 = wind speed [m/s] referenced to 10m, neutral stability as (m,) numpy.array with m>=1
    #
    # Optional Inputs (required for some sssf-parametrisations):
    # SST = sea surface temperature [C] as (m,) numpy.array with m>=1
    # Re = Reynolds number [1] Re=u*Hs/v_w as (m,) numpy.array with m>=1
    #
    # Output:
    # dFdr80, FN:
    # dF/dr80 [#/m^2/s/um ] production flux per size bin as numpy.array (m,n) or (n,) if m=1
    # FN [#/m^2/s] production flux integrated over range of the r80 provided (r80[0] to r80[-1]) as numpy.array (m,)
    #
    # to convert form dF/dr80 to dF/d(log(r80)) using d(ln(x))/dx = 1/x and d(log10(x))/dx = log10(e)*1/x
    # -> dF/r80 = dF/d(log10(r80))*d(log10(r80))/dr80 = dF/d(log10(r80)) * log10(e)* (1/x)
    #
    # if we speek of dF/dlog(r80) below we mean dF/dlog10(r80) otherwise we write dF/dln(r80)
    # note in python np.log = ln!

    U10 = U10.reshape(len(U10),1)

    if sssf_str in ['Jaeg11', 'Jaegele 2011', 'Jaegele et al., 2011']:
        SST = SST.reshape(len(U10),1) # this way allows inputting SST as single value
        SST[SST<0]=0 # otherwise the results get negative
        #'Jaegele et al., 2011' eq. 4 (SST-model) based on Gong03
        Theta=30 # tunig parameter introduced by Gong03 to fit submicron particles suggesting Theta=30
        # range: r80=0.07-20um
        B=(0.433-np.log10(r80))/0.433;
        A=4.7*np.power(1+Theta*r80, -0.017*np.power(r80,-1.44) )
        fT=(0.3+0.1*SST - 0.0076*SST*SST + 0.00021*SST*SST*SST)
        dFdr80 = fT*1.373*np.power(U10,3.41)*np.power(r80,-A)*(1+0.057*np.power(r80,3.45))*np.power(10,(1.607*np.exp(-np.power(B,2)) ))
        dFdlogr80 = dFdr80*r80/np.log10(np.exp(1))

    if sssf_str == 'Gong03':
        #Gong03
        #Monahan et al. (1986):
        Theta=30 # tunig parameter introduced by Gong03 to fit submicron particles suggesting Theta=30
        # range: r80=0.07-20um
        B=(0.433-np.log10(r80))/0.433;
        A=4.7*np.power(1+Theta*r80, -0.017*np.power(r80,-1.44) )
        dFdr80 = 1.373*np.power(U10,3.41)*np.power(r80,-A)*(1+0.057*np.power(r80,3.45))*np.power(10,(1.607*np.exp(-np.power(B,2)) ))
        dFdlogr80 = dFdr80*r80/np.log10(np.exp(1))

    if sssf_str == 'MM86':
        #MM86
        #Monahan et al. (1986):
        # Wcap method + lab experiments: range: r80=0.8-8um
        B=(0.380-np.log10(r80))/0.650;
        #W=3.84*1E-6*np.power(U10,3.41); # WhitCap fraction from Monahan and O'Muirchaetaigh 1980
        # from Gyrte2017?? dFdr80 = W*3.6*1E5*np.power(r80,-3)*(1+0.057*np.power(r80,1.05))*np.power(10,(1.19*np.exp(-np.power(B,2)) ))
        # from Gong03:
        dFdr80 = 1.373*np.power(U10,3.41)*np.power(r80,-3)*(1+0.057*np.power(r80,1.05))*np.power(10,(1.19*np.exp(-np.power(B,2)) ))
        dFdlogr80 = dFdr80*r80/np.log10(np.exp(1))

    elif sssf_str == 'LS04':
        # Either de Leeuw plots it wrong or he is missing a np.log10(np.exp(1))?
        # cause it does not match with the  'LS04wet' on the plot
        # from de Leeuw 2011 [171]: claimed to be Lewis and Schwarz based on multiple methods (can't find it in L&S!)
        # range: r80=0.1-25um; U10=5-20m/s
        dFdlogr80 = 50*np.power(U10,2.5)*np.exp( -0.5*np.power( np.log(r80/0.3)/np.log(4) ,2) )
        dFdr80 = dFdlogr80/r80*np.log10(np.exp(1)) # added np.log10(np.exp(1)) as the origial is dFdlog10/dr80

    elif sssf_str == 'LS04wet':
        # from de Leeuw 2011 [171]: claimed to be Lewis and Schwarz based on multiple methods
        # range: r80=0.1-25um; U10=5-20m/s

        dFdlogr80 = 1E4*np.ones_like(r80)*np.ones_like(U10)
        dFdr80 = dFdlogr80/r80*np.log10(np.exp(1)) # added np.log10(np.exp(1)) as the origial is dFdlog10/dr80
        # below not veryfied! / finished
    elif sssf_str == 'Ma03':
        # Martenson 2003 from de Leeuw 2011 [163] and table A1:
        #
        1+1
    elif sssf_str in ['Ov14', 'Ovadnevaite 2014', 'Ovadnevaite et al., 2014']:
        Re = Re.reshape(len(Re),1)

        sigma_i = [1.37, 1.5, 1.42, 1.53, 1.85]
        CMD_i = [0.018, 0.041, 0.09, 0.23, 0.83]
        A_FiRe = [104.5, 0.0442, 149.6, 2.96, 0.51]
        C_FiRe = np.array([-1, -1, -1, -1, -2])*1E-5
        B_FiRe = [0.556, 1.08, 0.545, 0.79, 0.87]

        df_FiRe = pd.DataFrame({'sigma_i': sigma_i, 'CMD_i': CMD_i, 'A_FiRe': A_FiRe, 'B_FiRe': B_FiRe, 'C_FiRe': C_FiRe})

        dFdlogr80 = np.zeros([len(Re),len(r80)])
        for j in df_FiRe.index:
            FiRe = df_FiRe['A_FiRe'][j]*np.power((Re + df_FiRe['C_FiRe'][j]),df_FiRe['B_FiRe'][j])
            # below Eq. 2 which calcualtes dFdlnD_i
            dFdlnD_i = FiRe/np.sqrt(2*np.pi)/np.log(df_FiRe['sigma_i'][j])*np.exp(-0.5 * np.power(np.log(r80/df_FiRe['CMD_i'][j])/np.log(df_FiRe['sigma_i'][j]),2) )
            # I added the 1/np.log10(np.exp(1)) here cause I assume that (2) gives dF/dln(D):
            dFdlogr80 = dFdlogr80+dFdlnD_i/np.log10(np.exp(1))

        dFdr80 = dFdlogr80/r80*np.log10(np.exp(1)) #

    # for calculating mass fluxes
    #rho_ss = 2.2*np.power(10.,-3*(6-2)) # g/cm^3 -> g/um^3
    #dFmdr80 = dFdr80*( 1/8*4*np.pi/3*rho_ss*np.power(r80,3) ) # 1/8=(rdry/r80)^3

    #FM = np.trapz(dFmdr80,x=r80) # integrate over given range to get the mass flux [g/m^2/s]
    FN = np.trapz(dFdr80,x=r80) # integrate over given range to get the number flux [#/m^2/s]
    dFdr80 = dFdr80.squeeze() # reduce unnecessary dimensions, [#/m^2/s/um]

    # dFdr80.shape = (m,n) or (n,) if m=1 not sure if I should modify this ?
    # FN.shape() = (m,)
    return dFdr80, FN

def aps_DlowDhigh_to_range(Dca_l,Dca_h,RESOLUTION=1/32):
    Dca_l = np.power(10,np.log10(Dca_l)-RESOLUTION/2) # 1/2 logarithmic step down
    Dca_h = np.power(10,np.log10(Dca_h)+RESOLUTION/2) # 1/2 logarithmic step up
    return np.array([Dca_l, Dca_h])

def aps_D_to_Dphys(Dca, rhop=2.017, chic=1.08):
    # converts APS aerodynamic diamter into physical diameter
    # assume continous flow regime => cunningham slip factors ~1
    # ρ0 = 1g/cm^3
    # ρp = 2.017 g/cm3 = sea salt density (Ziegler 2017)
    # χ_c = 1.08 (shape factor for cubic shape)
    # Dve volume equivalent diameter of the dried sea salt particle (assume this equals the physical diameter)
    # Dve = Dca √(χ_c  ρ0/ρp)
    Dve = Dca*np.sqrt(chic*1.0/rhop)
    return Dve



def aps_D_to_r80(Dca, rhop=2.017, chic=1.08, gf=2):
    # assume continous flow regime => cunningham slip factors ~1
    # ρ0 = 1g/cm^3
    # ρp = 2.2g/cm^3 (sea salt) -> changed to 2.017 g/cm3 (Ziegler 2017)
    # χ_c = 1.08 (cubic shape)
    # gf = hygroscopic growth factor: use gf=r_div_r80(RH)
    # Dve volume equivalent diameter of the dried sea salt particle (assume this equals r80)
    # Dve = Dca √(χ_c  ρ0/ρp)
    Dve = Dca*np.sqrt(chic*1.0/rhop)
    r80=Dve*gf/2
    return r80

def aps_aggregate(APS,AGG_WINDOWS, label_prefix='APS_', LABELS=[]):
    # FOR NOW I ASSUME THAT:
    # diameters given is center diameters of the logarithmic intervals, this would make sense cause:
    #1/(np.log10(0.523)-np.log10(0.542)) #-> 64
    #1/(np.log10(20.5353)-np.log10(19.81)) #-> 64
    #1/(np.log10(.523)-np.log10(0.486968)) #-> 32
    # all other ->32
    #(but may well be that they define the edges), recheck at some point!
    #
    # DOES IT MAKE MORE SENSE TO TIME AVERAGE then AGGREGATE OR wise versa??? ---> how to deal with 0 values?
    #
    ##Lower Channel Bound,0.486968
    ##Upper Channel Bound,20.5353
    ##Sample #,Date,Start Time,Aerodynamic Diameter,<0.523,0.542,0.583,0.626,0.673,0.723,0.777,0.835,0.898,0.965,1.037,1.114,1.197,1.286,1.382,1.486,1.596,1.715,1.843,1.981,2.129,2.288,2.458,2.642,2.839,3.051,3.278,3.523,3.786,4.068,4.371,4.698,5.048,5.425,5.829,6.264,6.732,7.234,7.774,8.354,8.977,9.647,10.37,11.14,11.97,12.86,13.82,14.86,15.96,17.15,18.43,19.81,Event 1,Event 3,Event 4,Dead Time,Inlet Pressure,Total Flow,Sheath Flow,Analog Input Voltage 0,Analog Input Voltage 1,Digital Input Level 0,Digital Input Level 1,Digital Input Level 2,Laser Power,Laser Current,Sheath Pump Voltage,Total Pump Voltage,Box Temperature,Avalanch Photo Diode Temperature,Avalanch Photo Diode Voltage,Status Flags,Median(µm),Mean(µm),Geo. Mean(µm),Mode(µm),Geo. Std. Dev.,Total Conc.
    ##1,04/08/17,06:54:36,dN/dlogDp,17.2413,7.51345,8.47983,9.01102,9.59981,9.40141,10.2206,13.7661,16.0893,16.0637,15.2573,13.9069,12.4926,11.8078,11.2382,10.195,8.94062,8.01904,7.60945,6.77106,6.18228,5.76628,4.8511,3.30873,2.13116,1.44637,0.684786,0.582388,0.275194,0.179196,0.179196,0.0831983,0.0191996,0.0319994,0.0255995,0.00639987,0,0.0127997,0.0127997,0.0127997,0.0191996,0,0.00639987,0.0127997,0.00639987,0.00639987,0.0127997,0,0,0.00639987,0,0,13860,78,21,187,999,4.97,3.98,0,0,0,0,0,75,66.7,3.079,3.422,32.5,29.5,211,0000 0000 0000 0000,1.07946,1.25393,1.13275,0.897687,1.55022,11.568(#/cm³)
    # 52 channels, but first one dropped by Julia
    # 02_diameterforfile_01.txt:
    # 0.542,  0.583,  0.626,  0.673,  0.723,  0.777,  0.835,  0.898, 0.965,  1.037,  1.114,  1.197,  1.286,  1.382,  1.486,  1.596, 1.715,  1.843,  1.981,  2.129,  2.288,  2.458,  2.642,  2.839, 3.051,  3.278,  3.523,  3.786,  4.068,  4.371,  4.698,  5.048, 5.425,  5.829,  6.264,  6.732,  7.234,  7.774,  8.354,  8.977, 9.647, 10.37 , 11.14 , 11.97 , 12.86 , 13.82 , 14.86 , 15.96 , 17.15 , 18.43 , 19.81
    # data from column j is from cloumn_header(j-1) till j

    from collections import defaultdict

    APS[(APS == 0).sum(axis=1)==len(APS.columns.values)]=np.nan # some rows are all zero -> set them to nan
    part_legend= np.array(list(map(float, APS.columns.values)))

    # maybe turn below into a function
    aps_agg = pd.DataFrame()
    aps_agg_meta = defaultdict(list)

    #APS scale of dN/dlogDp
    #plt.plot( 1/(np.log10(part_legend[1:]) - np.log10(part_legend[0:-1])) )
    aps_scale = 1/np.mean((np.log10(part_legend[1:]) - np.log10(part_legend[0:-1])))
    print(aps_scale)

    jlabel = -1
    for AGG_WINDOW in AGG_WINDOWS:
        print(AGG_WINDOW)
        agg_str = label_prefix+str(round(AGG_WINDOW[0]*1000))+'_'+str(round(AGG_WINDOW[1]*1000)) # give in nm to avoid the .
        if len(LABELS)==len(AGG_WINDOWS):
            jlabel=jlabel+1
            agg_str = label_prefix+LABELS[jlabel] # custom label set by user

        cond = (part_legend >= AGG_WINDOW[0]) & (part_legend<AGG_WINDOW[1])
        aps_agg = aps_agg.assign(newcol=APS.iloc[:,np.where(cond)[0]].sum(axis=1)/aps_scale)
        # now set to nan where we have no data (could be more strict by requesting all sizebins with cond==True to be present, for APS does not matter)
        aps_agg['newcol'].loc[np.sum(APS.iloc[:,np.where(cond)[0]].isnull(),axis=1) == cond.sum()] = np.nan
        #aps_agg['newcol'][(APS == 0).sum(axis=1)==51]=np.nan #

        aps_agg = aps_agg.rename(columns={'newcol': agg_str})

        aps_agg_meta['label'].append(agg_str)
        aps_agg_meta['Dp_aps_low'].append(part_legend[np.where(part_legend >= AGG_WINDOW[0])[0][0]])
        aps_agg_meta['Dp_aps_high'].append(part_legend[np.where(part_legend < AGG_WINDOW[1])[0][-1]])

    aps_agg_meta = pd.DataFrame(aps_agg_meta)

    return aps_agg, aps_agg_meta

def merge_wind_wave_parameters(SST_from='merge_ship_satellite', TA_from='ship', WAVE_from='imu',
    MET_DATA='../data/intermediate/0_shipdata/metdata_5min_parsed.csv',
    ERA5_DATA='../../ecmwf-download/data/ecmwf-on-track/era5_ace_track_nearest.csv',
    WIND_DATA='../data/intermediate/0_shipdata/u10_ship_5min_full_parsed.csv',
    WAVE_DATA='../data/intermediate/17_waves/01_waves_recomputed.csv',
    WAMOS_DATA='../data/intermediate/17_waves/Updated_Wave_Info_ACE_Leg01234_parsed.csv',
    IMU_DATA='../data/intermediate/17_waves/WaveInfo_ACE_IMU_parsed.csv',
    FERRYBOX='../data/intermediate/18_percipitation/ferrybox_parsed.csv',
    SATELLITE='../data/intermediate/18_percipitation/satellite_parsed.csv',
    SWI_DATA='../data/intermediate/11_meteo/ACE_watervapour_isotopes_SWI13_5min_parsed.csv',
    D_TO_LAND='../data/intermediate/0_shipdata/BOAT_GPS_distance_to_land_parsed.csv',
    T_TO_LAND='../data/intermediate/7_aerosols/hours_till_land_parsed.csv',
    RETURN_EXPANSIONS=False):

    # loads parameters from various data sets and creates a joint 5min data set
    # the 5min time stamp label resides on the LEFT (start) of the 5min interval
    #
    #
    # this function uses ship based observations and era5 interpolated model output to fill gaps
    #
    # sst_from='era5' : use era5 sst
    # sst_from='ship' : use ferrybox temperature (+273.15 to convert to Kelvin)
    # sst_from='merge', 'merge_ship_satellite': merge satellite and ferrybox temperature
    # sst_from='merge_ship_era5': merge era5 and ferrybox temperature
    #
    # TA_from='ship'  : use metdata 'TA' (+273.15 to convert to Kelvin)
    #
    # WAVE_from='imu' : use IMU for total sea, wind and swell are set to NaN
    # WAVE_from='wamos' : use Wamos data, total, wind and swell separately
    # I would not trust air temperature from the Mode
    #

    MET_DATA  = Path(MET_DATA)
    ERA5_DATA = Path(ERA5_DATA)
    # era5 linear, provides NaN if one of neighboring fields is land (lsm=1)
    # era5 nearest, provides NaN if the nearest fields is land (lsm=1)
    WIND_DATA = Path(WIND_DATA)
    WAVE_DATA = Path(WAVE_DATA)
    WAMOS_DATA = Path(WAMOS_DATA)
    IMU_DATA = Path(IMU_DATA)
    
    FERRYBOX = Path(FERRYBOX)
    SATELLITE = Path(SATELLITE)
    SWI_DATA = Path(SWI_DATA)

    # Buffers to land: distance to any land and time to any land

    D_TO_LAND = Path(D_TO_LAND)
    T_TO_LAND = Path(T_TO_LAND)

    metdata = dataset.read_standard_dataframe(MET_DATA, crop_legs=False)
    if (str(metdata.index.tzinfo)=='None')==False:
        metdata.set_index(metdata.index.tz_convert(None), inplace=True) # remove TZ info
    metdata.index = metdata.index-pd.Timedelta(5,'min')

    wind = dataset.read_standard_dataframe(WIND_DATA, crop_legs=False)
    if (str(wind.index.tzinfo)=='None')==False:
        wind.set_index(wind.index.tz_convert(None), inplace=True) # remove TZ info
    wind.index = wind.index-pd.Timedelta(5,'min')

    wind = pd.merge(wind,metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
    wind.drop(columns=['VIS'], inplace=True)

    d_to_land = dataset.read_standard_dataframe(D_TO_LAND, crop_legs=False)
    d_to_land.index = d_to_land.index-pd.Timedelta(5,'min')
    if (len(d_to_land.columns)>1): # if intermediate version is used the csv also contains lat/lon, need to drop these
        d_to_land.drop(columns=['latitude', 'longitude'], inplace=True)

    t_to_land = dataset.read_standard_dataframe(T_TO_LAND, crop_legs=False)
    t_to_land.index = t_to_land.index-pd.Timedelta(5,'min')
    t_to_land = pd.merge(t_to_land[['hours_till_land']],metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
    t_to_land.drop(columns=['VIS'], inplace=True) # drop the unnecessary column we got from merging
    t_to_land = t_to_land.interpolate(axis=0, method='nearest', limit=20, limit_direction='both') # interpolate between the 1 hourly data points

    era5 = dataset.read_standard_dataframe(ERA5_DATA, crop_legs=False)
    if (str(era5.index.tzinfo)=='None')==False:
        era5.set_index(era5.index.tz_convert(None), inplace=True) # remove TZ info
    era5.index = era5.index-pd.Timedelta(2.5,'min') # adjust to beginning of 5min interval rule
    era5 = pd.merge(era5,metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
    era5.drop(columns=['VIS'], inplace=True)
    
    satellite = dataset.read_standard_dataframe(SATELLITE, crop_legs=False)
    
    satellite = dataset.resample_timeseries(satellite, time_bin=5, how='mean', new_label_pos='l', new_label_parity='even', old_label_pos='c', old_resolution=60) # old resolution = 60min but time stamp irregular on odd seconds -> resample to 5min
    satellite = dataset.match2series(satellite,metdata) # match to params seris
    satellite = satellite.interpolate(axis=0, method='nearest', limit=20, limit_direction='both') # interpolate between the 1 hourly data points
       
    ferrybox = dataset.read_standard_dataframe(FERRYBOX, crop_legs=False)
    ferrybox = ferrybox.resample('5min' ).mean() # resample 5min
    ferrybox = dataset.match2series(ferrybox,metdata) # match to params seris


    params = wind[['u10', 'uR_afc', 'vR_afc']].copy() # 10meter neutral wind speed [m/s] and relative wind vector
    params['WSR_afc'] = np.sqrt(np.square(params['uR_afc'])+np.square(params['vR_afc']))
    params['WDR_afc'] = (180 - np.rad2deg(np.arctan2(params['vR_afc'],params['uR_afc']) ) )%360 #
    params['visibility'] = metdata['VIS']
    
    params['d-to-land'] = d_to_land
    params['t-to-land'] = t_to_land
    # interplate accross the gaps in d-to-land which are due to missing gps.
    for var_str in ['d-to-land', 't-to-land']:
        params[var_str] = params[var_str].interpolate(method='linear', limit=200, limit_direction='both', axis=0)
    
    params['RH'] = metdata['RH'] # relative humidity [%]
    params['TA'] = metdata['TA']+273.15 # air temperature [K] #90 5min data points are NaN & (u10~NaN and LSM==0)
    params['PA'] = metdata['PA'] # atmospheric pressure in hPa = mbar

    if 0: # interpolate over small gaps in the TA, RH
        for var_str in ['RH', 'TA']:
            params[var_str] = params[var_str].interpolate(method='linear', limit=4, limit_direction='both', axis=0)
   
    
    if SST_from in ['merge', 'ship', 'merge_ship_satellite', 'merge_ship_era5']:
        params['SST'] = ferrybox['temperature']+273.15
        
        if SST_from in ['merge', 'merge_ship_satellite']:
            # fill large gaps with satellite SST
            FILLGAPS = (( np.isnan(ferrybox.temperature) & np.isnan(ferrybox.interpolate(method='linear', limit=12, limit_direction='both', axis=0)['temperature'])   ))
            params['SST'][FILLGAPS] = satellite.sst[FILLGAPS]+273.15
        if SST_from in ['merge_ship_era5']:
            # fill the large gaps with era5 sst (-273.15),
            # there are local mismatches between era5 and ferrybox.temperature,
            # in oder to avoid plenty of jumpy data we only fill gaps that are longer than 1hour=12*5min in either direction:
            FILLGAPS = (( np.isnan(ferrybox.temperature) & np.isnan(ferrybox.interpolate(method='linear', limit=12, limit_direction='both', axis=0)['temperature'])   ))
            params['SST'][FILLGAPS] = era5.sst[FILLGAPS]
        
    elif SST_from in ['era5']:
        params['SST'] = (era5['sst'])
    else:
        print('wrong option for SST_from, use: ship, era5, or merge')
        return
        
    if 1: # interpolate over 1hour gaps in the SST
        params['SST'] = params['SST'].interpolate(method='linear', limit=24, limit_direction='both', axis=0)
        
    params['deltaT']=(params['TA']-params['SST']) # air sea temperature gradient [K]

    params['BLH']=era5['blh'] # boundary layer height [m]

    kin_visc_sea = aceairsea.kinematic_viscosity_sea((params['SST']-273.15),35)
    params['ustar'] = aceairsea.coare_u2ustar (params['u10'], input_string='u2ustar', coare_version='coare3.5', TairC=20.0, z=10.0, zeta=0.0)
    
    # stable water isotopes
    swi = dataset.read_standard_dataframe(SWI_DATA,crop_legs=False)
    swi = dataset.resample_timeseries(swi, time_bin=5, how='mean', new_label_pos='l', new_label_parity='even', old_label_pos='r', old_resolution=5)
    swi = dataset.match2series(swi,params)
    for var_str in ['d18O','d2H','dexc']:
        params[var_str]=swi[var_str]

    # Wave derived parms:

    # wave time stamp already on left
    wave = dataset.read_standard_dataframe(WAVE_DATA)
    wave = pd.merge(wave,metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
    wave.drop(columns=['VIS'], inplace=True)
    #
    # define if to rename wave variables when writing to params or not!!!
    # params['what you like']=wave['wind_sea_hs']
    for var_str in ['total_age', 'total_hs', 'total_steep','total_tp',
                    'wind_sea_age', 'wind_sea_hs', 'wind_sea_steep', 'wind_sea_tp',
                    ]:
        params[var_str]=wave[var_str]

    wamos = dataset.read_standard_dataframe(WAMOS_DATA)
    wamos = wamos.resample('5min' ).mean() # resample to 5min

    imu = dataset.read_standard_dataframe(IMU_DATA)
    # remove Hs spikes and Tp data out of range
    X_out, _, _ = datafilter.outliers_iqr_time_window(imu[['Hs (m)']].copy(),Nmin=6*60,minN=7,d_phys=0.5, d_iqr=1.5) # perfect for Hs
    Tp_out = ((imu['Tp (s)']<4.5)| (imu['Tp (s)']>14))
    for var_str in imu.columns:
        imu.at[X_out,var_str]=np.NaN
        imu.at[Tp_out,var_str]=np.NaN

    imu = dataset.resample_timeseries(imu, time_bin=5, how='mean', new_label_pos='l', new_label_parity='even', old_label_pos='c', old_resolution=20) # old resolution = 20min but time stamp irregular on odd seconds -> resample to 5min
    imu = dataset.match2series(imu,params) # match to params seris
    #imu = imu.interpolate(axis=0, method='nearest', limit=2, limit_direction='both') # fill in nearest 2 5min blocks with the same 20min data
    imu = imu.interpolate(axis=0, method='nearest', limit=3, limit_direction='both') # allo to fill the neigbourblock


    if WAVE_from=='imu':   
        params['total_hs']=imu['Hs (m)']
        params['total_tp']=imu['Tp (s)']
        #params['total_steep']=imu['Steepness (-)']
        for var_str in ['wind_sea', 'swell']:
            params[var_str+'_hs']=imu['Hs (m)']*np.NaN
            params[var_str+'_tp']=imu['Hs (m)']*np.NaN

    elif WAVE_from=='wamos':
        for var_str in ['total', 'wind_sea', 'swell']:
            if var_str=='total': 
                wamos_str = 'Total'
            elif var_str=='wind_sea': 
                wamos_str = 'Wind Sea'
            else: 
                wamos_str = 'Swell'
            params[var_str+'_hs']=wamos[wamos_str+' Hs (m)']
            params[var_str+'_tp']=wamos[wamos_str+' Tp (s)']
    # recomputation of parameters
    # steepness, ReHs, LenainMelville, all from Tp and Hs
    # Wave number: kp=(2*pi/tp)^2/9.81
    # Steepness: st = 0.5*Hs*kp = 0.5*Hs*(2*pi/tp)^2/9.81
    # Phase velocity: Cp=Tp*g/2/pi
    # Age: Cp/U10 ; Cp/ustar
    # hs^1.25 g^0.5 kp^{-0.25} nu_w^{-1} # Lenain and Melville
    # kp=((4*pi / tp)^2)/g # wave number at peak frequency (total sea!)
    for var_str in ['total', 'wind_sea', 'swell']:
        params[var_str+'_kp']=4*np.pi*np.pi/params[var_str+'_tp']/params[var_str+'_tp']/9.81
        params[var_str+'_steep']=0.5*params[var_str+'_hs']*params[var_str+'_kp']
        params[var_str+'_age'] = params[var_str+'_tp']*9.81/2/np.pi/params['u10']
        params[var_str+'_ReHs'] = params['ustar']*params[var_str+'_hs']/kin_visc_sea
        params[var_str+'_LenainMelville'] = np.power(params[var_str+'_hs'],1.25)*np.power(9.81,.5)*np.power(params[var_str+'_kp'],-0.25)/kin_visc_sea

    


    if RETURN_EXPANSIONS:
        # add some expansions of u10, RH, and deltaT
        params['u10^2'] = np.power(params['u10'],2)
        params['u10^3'] = np.power(params['u10'],3)

        params['RH^2'] = np.power(params['RH'],2)
        params['RH^3'] = np.power(params['RH'],3)
        params['RH^4'] = np.power(params['RH'],4)

        params['deltaT^2'] = np.power(params['deltaT'],2)
        params['deltaT^3'] = np.power(params['deltaT'],3)
        params['deltaT^4'] = np.power(params['deltaT'],4)


    return params

def filter_parameters(data, d_lim=10000, t_lim=24, not_to_mask=1,  D_TO_LAND='../data/intermediate/0_shipdata/BOAT_GPS_distance_to_land_parsed.csv', T_TO_LAND='../data/intermediate/7_aerosols/hours_till_land_parsed.csv', MASK= '../data/intermediate/7_aerosols/mask_1_5_10min++_parsed.csv'):
    """
        filters parameters based on `time to land`, `distance to land` (either already contained or added to data) and a binary mask (1=keep)
        returns the filetered dataframe and the boolean array of VALID data points.

        Return
        ======

        data_filt : filtered data
        merged_conditions: conditions met, independently for each filtering attribute
    """
    D_TO_LAND = Path(D_TO_LAND); T_TO_LAND = Path(T_TO_LAND); MASK = Path(MASK)
    d_to_land_FLAG = False # flag to remove the column after filtering to return original data frame without additional columns
    t_to_land_FLAG = False
    if 'd-to-land' not in data.columns.tolist():
        d_to_land_FLAG = True
        d_to_land = dataset.read_standard_dataframe(D_TO_LAND, crop_legs=False)
        d_to_land.index = d_to_land.index-pd.Timedelta(5,'min')
        #data['d-to-land'] = d_to_land this does not work
        data = data.merge(d_to_land[['distance']],left_index = True, right_index=True, how='left')
        data.rename(columns={"distance": "d-to-land"}, inplace=True)
        # interplate accross the gaps in d-to-land which are due to missing gps.
        data['d-to-land'] = data['d-to-land'].interpolate(method='linear', limit=1000, limit_direction='both', axis=0)


    if 't-to-land' not in data.columns.tolist():
        t_to_land_FLAG = True
        t_to_land = dataset.read_standard_dataframe(T_TO_LAND, crop_legs=False)
        t_to_land.index = t_to_land.index-pd.Timedelta(5,'min')
        # here we first merge with data to cover the same range of 5min samples
        # then cause t_to_land is a 1hr time series we need to interpolate over the gaps.
        data = data.merge(t_to_land[['hours_till_land']],left_index = True, right_index=True, how='left')
        data.rename(columns={"hours_till_land": "t-to-land"}, inplace=True)
        data['t-to-land'] = data['t-to-land'].interpolate(axis=0, method='nearest', limit=20, limit_direction='both') # interpolate between the 1 hourly data points

        #data['t-to-land'] = t_to_land

    mask = dataset.read_standard_dataframe(MASK, crop_legs=False)
    mask.index = mask.index-pd.Timedelta(5,'min')
    mask = mask['mask_5min']
    data = data.merge(mask,left_index = True, right_index=True)

    keep_criterion = pd.DataFrame()
    keep_criterion['d-to-land'] = (data['d-to-land'] <= d_lim)
    keep_criterion['t-to-land'] = (data['t-to-land'] <= t_lim)
    keep_criterion['mask'] = (data['mask_5min'] != not_to_mask)

    keep_in = (data['d-to-land'] > d_lim) & (data['t-to-land'] > t_lim) & (data['mask_5min'] == not_to_mask)

    data_filt = data.copy()
    #print(data_filt.columns) #  [..., 'd-to-land', 't-to-land', 'mask_5min']
    #data_filt.loc[~keep_in,:-3] = np.nan
    # set all the data to nan, but leaf 'd-to-land', 't-to-land', 'mask_5min' with their values
    for col in data_filt.columns:
        if col not in list(['d-to-land', 't-to-land', 'mask_5min']):
            data_filt.loc[~keep_in,col] = np.nan
    data_filt = data_filt.drop('mask_5min',axis=1)
    
    # if d-to-land, t-to-land were not in the data frame drop them again
    if d_to_land_FLAG:
        data_filt = data_filt.drop('d-to-land',axis=1)
    if t_to_land_FLAG:
        data_filt = data_filt.drop('t-to-land',axis=1)
    return data_filt, keep_criterion


    