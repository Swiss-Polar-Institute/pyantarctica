# some functions needed in the sea spray source function project
import numpy as np
import pandas as pd
from pathlib import Path
import pyantarctica.aceairsea as aceairsea
import pyantarctica.dataset as dataset


def r_div_r80(RH, option='Lewis2004'):
    # hygroscopic growth factor
    # (below RH=75% this is valid only for decreasing RH)!
    # THIS NEGLECTS THE KELVIN EFFECT!
    # But the difference should be really small
    if option == 'Lewis2004':
        # within 2.5% of Tang 1997 for RH>50%,
        r_div_r80 = 0.54*np.power((1+1/(1-RH/100)), 1/3)
    r_div_r80[r_div_r80>2]=2 # limit to 98%?-> gf=2.
    r_div_r80[RH<42]=0.5 # for low RH set rRH=rDry=0.5*r80
    return r_div_r80


def rdry2rRH(Dp, RH, option='Lewis2004'):
    # convert try diameter/radius to expected diameter/radius at RH[%]
    # rRH/rDry = (rRH/r80)*(r80/rDry) = (rRH/r80)*2
    return 2*r_div_r80(RH, option=option)*Dp

def dFdlogD_to_dFdD(dFdlogr80,r80):
    dFdr80 = dFdlogr80/r80*np.log10(np.exp(1))
    return dFdr80

def dFdD_to_dFdlogD(dFdr80,r80):
    dFdlogr80 = dFdr80*r80/np.log10(np.exp(1))
    return dFdlogr80

#def deposition_velocity(model_str, Dp, rho_p, h_ref, U10, T, P, zeta=0):
def deposition_velocity(Dp, rho_p=2.2, h_ref=15., U10=10., T=20, P=1013., zeta=0., model='giardina_2018'):

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
    # vd = deposition velocity [m/s ] production flux per size bin as numpy.array (m,n)
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
    rho_p = rho_p.reshape(len(rho_p),1)

    if ((len(U10)==len(Dp)) & (len(Dp)>1) ):
        # in this case assume each Dp sample corresponds to a U10 sample and we get a time series of vd
        Dp = Dp.reshape(len(Dp),1) #
    #if ((len(U10)==len(rho_p)) & (len(rho_p)>1) ):
    #    # in this case assume each Dp sample corresponds to a U10 sample and we get a time series of vd
    #    rho_p = rho_p.reshape(len(rho_p),1) #


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

    # Diffusivity
    Diffusivity = kBolz*T*Ccunn/3/np.pi/dyn_visc/Dp # Diffusivity of the aerosol in air

    # setttling velocity in m/s,note that the Dp and rho_p play an important role!
    vs = g*rho_p*Dp*Dp*Ccunn/18/dyn_visc


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

def merge_wind_wave_parameters(SST_from='era5', TA_from='ship',
    MET_DATA='../data/intermediate/0_shipdata/metdata_5min_parsed.csv',
    ERA5_DATA='../../ecmwf-download/data/ecmwf-on-track/era5_ace_track_nearest.csv',
    WIND_DATA='../data/intermediate/0_shipdata/u10_ship_5min_full_parsed.csv',
    WAVE_DATA='../data/processed/17_waves/01_waves_recomputed.csv',
    D_TO_LAND='../data/processed/0_shipdata/BOAT_GPS_distance_to_land_parsed.csv',
    T_TO_LAND='../data/processed/7_aerosols/hours_till_land_parsed.csv'):

    # TODO check time stamp label for wave data
    # add bt_time till land and distance to (populated)land

    # load intermediate data and compute parameters for the aerosol wind wave regression
    #
    # this function uses ship based observations and era5 interpolated model output to fill gaps

    # sst_from='era5' : use era5 sst
    # sst_from='ship' : use ferrybox temperature (+273.15 to convert to Kelvin)
    # sst_from='merge': merge era5 and ferrybox temperature

    # TA_from='ship'  : use metdata 'TA' (+273.15 to convert to Kelvin)
    # I would not trust air temperature from the Mode
    MET_DATA  = Path(MET_DATA)
    ERA5_DATA = Path(ERA5_DATA)
    # era5 linear, provides NaN if one of neighboring fields is land (lsm=1)
    # era5 nearest, provides NaN if the nearest fields is land (lsm=1)
    WIND_DATA = Path(WIND_DATA)
    WAVE_DATA = Path(WAVE_DATA)
    # Buffers to land: distance to any land and time to any land

    D_TO_LAND = Path(D_TO_LAND)
    T_TO_LAND = Path(T_TO_LAND)

    metdata = dataset.read_standard_dataframe(MET_DATA, crop_legs=False)
    metdata.index = metdata.index-pd.Timedelta(5,'min')

    wind = dataset.read_standard_dataframe(WIND_DATA, crop_legs=False)
    wind.index = wind.index-pd.Timedelta(5,'min')

    wind = pd.merge(wind,metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
    wind.drop(columns=['VIS'], inplace=True)

    d_to_land = dataset.read_standard_dataframe(D_TO_LAND, crop_legs=False)
    d_to_land.index = d_to_land.index-pd.Timedelta(5,'min')

    t_to_land = dataset.read_standard_dataframe(T_TO_LAND, crop_legs=False)
    t_to_land.index = t_to_land.index-pd.Timedelta(5,'min')

    era5 = dataset.read_standard_dataframe(ERA5_DATA, crop_legs=False)
    era5.set_index(era5.index.tz_convert(None), inplace=True) # remove TZ info
    era5.index = era5.index-pd.Timedelta(2.5,'min') # adjust to beginning of 5min interval rule
    era5 = pd.merge(era5,metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
    era5.drop(columns=['VIS'], inplace=True)

    if SST_from in ['merge', 'ship']:
        ferrybox_file_folder = os.path.join('..','..','local_data','ace_ferrybox_giuseppe')
        ferrybox = []
        for fbox_filename in ['track_ferrybox_data_20161220_20170118_1min.csv', 'track_ferrybox_data_20170122_20170223_1min.csv', 'track_ferrybox_data_20170226_20170319_1min.csv']:
            ferrybox.append(pd.read_csv(os.path.join(ferrybox_file_folder,fbox_filename), na_values=['-9.9999'] ))
        ferrybox = pd.concat(ferrybox)

        ferrybox = ferrybox.rename(index=str, columns={"date_time": "timest_"})
        ferrybox = ferrybox.set_index(pd.to_datetime(ferrybox.timest_, format="%Y-%m-%d %H:%M:%S")) # assing the time stamp
        ferrybox = ferrybox.drop(columns=['timest_'])
        ferrybox = ferrybox.resample('5min' ).mean() # resample 5min
        ferrybox = pd.merge(ferrybox,metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
        ferrybox.drop(columns=['VIS'], inplace=True)

    params = wind[['u10']].copy() # 10meter neutral wind speed [m/s]
    params['d-to-land'] = d_to_land
    params['t-to-land'] = t_to_land
    params['RH'] = metdata['RH'] # relative humidity [%]
    params['TA'] = metdata['TA']+273.15 # air temperature [K] #90 5min data points are NaN & (u10~NaN and LSM==0)

    if SST_from in ['merge', 'ship']:
        params['SST'] = ferrybox['temperature']+273.15
    elif SST_from in ['era5']:
        params['SST'] = (era5['sst'])
    else:
        print('wrong option for SST_from, use: ship, era5, or merge')
        return

    if SST_from in ['merge']:
        # fill the large gaps with era5 sst (-273.15),
        # there are local mismatches between era5 and ferrybox.temperature,
        # in oder to avoid plenty of jumpy data we only fill gaps that are longer than 1hour=12*5min in either direction:
        params['SST'][(( np.isnan(ferrybox.temperature) & np.isnan(ferrybox.interpolate(method='linear', limit=12, limit_direction='both', axis=0)['temperature'])   ))] = (era5.sst[(( np.isnan(ferrybox.temperature) & np.isnan(ferrybox.interpolate(method='linear', limit=12, limit_direction='both', axis=0)['temperature'])   ))])
    elif SST_from in ['ship', 'era5']:
        1+1 # nothing to do here
    else:
        print('wrong option for SST_from, use: ship, era5, or merge')
        return

    params['deltaT']=(params['TA']-params['SST']) # air sea temperature gradient [K]

    params['BLH']=era5['blh'] # boundary layer height [m]

    kin_visc_sea = aceairsea.kinematic_viscosity_sea((params['SST']-273.15),35)
    params['ustar'] = aceairsea.coare_u2ustar (params['u10'], input_string='u2ustar', coare_version='coare3.5', TairC=20.0, z=10.0, zeta=0.0)

    # Wave derived parms:

    # HOW IS THE TIME STAMP CONVENTION OF THIS ONE????
    # WHERE ARE THE CALCULTIONS DONE, I would like to check them
    wave = dataset.read_standard_dataframe(WAVE_DATA)
    wave = pd.merge(wave,metdata[['VIS']],left_index=True,right_index=True,how='right',suffixes=('', '')) # merge to get numbers right
    wave.drop(columns=['VIS'], inplace=True)
    #
    # define if to rename wave variables when writing to params or not!!!
    # params['what you like']=wave['wind_sea_hs']
    for var_str in ['total_age', 'total_hs', 'total_steep',
                    'wind_sea_age', 'wind_sea_hs', 'swell_steep']:
        params[var_str]=wave[var_str]

    # computation of reighnolds number for total sea and wind see
    params['wind_sea_ReHs'] = params['ustar']*wave['wind_sea_hs']/kin_visc_sea
    params['total_ReHs'] = params['ustar']*wave['total_hs']/kin_visc_sea

    # hs^1.25 g^0.5 kp^{-0.25} nu_w^{-1} # Lenain and Melville
    # kp=((4*pi / tp)^2)/g # wave number at peak frequency (total sea!)
    params['total_LenainMelville'] = np.power(wave['total_hs'],1.25)*np.power(9.81,.5)*np.power(wave['total_wave_number'],-0.25)/kin_visc_sea
    params['wind_sea_LenainMelville'] = np.power(wave['wind_sea_hs'],1.25)*np.power(9.81,.5)*np.power(wave['wind_sea_wave_number'],-0.25)/kin_visc_sea

    return params
