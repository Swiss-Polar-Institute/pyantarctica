# some functions needed in the sea spray source function project
import numpy as np
import pandas as pd
from collections import defaultdict


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
    # to convert form dF/dr80 to dF/d(log(r80)) using d(log(x))/dx = 1/x
    # dF/r80 = dF/d(log(r80))*d(log(r80))/dr80 = dF/d(log(r80)) * (1/x)
    #
    # 
    
    U10 = U10.reshape(len(U10),1)
    
    if sssf_str == 'Gong03':
        #Gong03
        #Monahan et al. (1986):
        Theta=30 # tunig parameter introduced by Gong03 to fit submicron particles suggesting Theta=30
        B=(0.433-np.log10(r80))/0.433;
        A=4.7*np.power(1+Theta*r80, -0.017*np.power(r80,-1.44) )
        dFdr80 = 1.373*np.power(U10,3.41)*np.power(r80,-A)*(1+0.057*np.power(r80,3.45))*np.power(10,(1.607*np.exp(-np.power(B,2)) ))
        dFdlogr80 = dFdr80*r80
    
    if sssf_str == 'MM86':
        #MM86
        #Monahan et al. (1986):
        B=(0.380-np.log10(r80))/0.650; 
        #W=3.84*1E-6*np.power(U10,3.41); # WhitCap fraction from Monahan and O'Muirchaetaigh 1980
        # from Gyrte2017?? dFdr80 = W*3.6*1E5*np.power(r80,-3)*(1+0.057*np.power(r80,1.05))*np.power(10,(1.19*np.exp(-np.power(B,2)) ))
        # from Gong03
        dFdr80 = 1.373*np.power(U10,3.41)*np.power(r80,-3)*(1+0.057*np.power(r80,1.05))*np.power(10,(1.19*np.exp(-np.power(B,2)) ))
        dFdlogr80 = dFdr80*r80

    # below not veryfied!
    elif sssf_str == 'LS04':
        #dFdr80 = 500*np.power(U10,2.5)*np.power(r80,-1.65) #version from Gyrte2017 appears to be wrong
        #dFdlogr80 = dFdr80*r80
        dFdlogr80 = 50*np.power(U10,2.5)*np.exp( -0.5*np.power( np.log(r80/0.3)/np.log(4) ,2) ) # from de Leeuw 2011
        dFdr80 = dFdlogr80/r80
        
    elif sssf_str == 'A07':
        #A07
        #Andreas (2007), a revised PP06:
        #using r80=r80
        r80[r80<.25]=np.nan
        dFdr80 = 0.4*np.exp((0.52*U10+0.64)*r80)
        dFdlogr80 = dFdr80*r80

    elif sssf_str == 'A90':
        #A90 
        #the source function given in Andreas (1990), as presented in Andreas et al. (1995) and L&S2004:
        # note this is F interface ! according to L&S need to fold with Fairall and Larsen 1984
        # supposedly applicable for 0.8um<r80<15um; U10<20m/s
        L = np.log10(r80) # L=log10(r80/um)
        dFdr80 = np.power(U10,2.22)*np.power(10, (2.4447-1.6784*L -2.4581*L*L+7.7635*L*L*L-3.9667*L*L*L*L) )
        dFdlogr80 = dFdr80*r80
        # equation from L&S and Grythe 2017

    elif sssf_str == 'PP06':
        #PP06 is from Petelski and Piskozub (2006) (applied as presented as in de Leeuw et al., 2011):
        dFdlogr80 = 70*np.exp(0.21*U10)*np.power(r80,3)*np.exp(-0.58*r80) / (1-np.exp(-0.11*r80*r80/U10) )        
        dFdr80 = dFdlogr80*r80

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
    
def aps_D_to_r80(Dca):
    # assume continous flow regime => cunningham slip factors ~1
    # ρ0 = 1g/cm^3
    # ρp = 2.2g/cm^3 (sea salt)
    # χ_c = 1.08 (cubic shape)
    # Dve volume equivalent diameter of the dried sea salt particle (assume this equals r80)
    # Dve = Dca √(χ_c  ρ0/ρp)
    Dve = Dca*np.sqrt(1.08*1.0/2.2)
    r80=Dve
    return r80
    
def aps_aggregate(APS,AGG_WINDOWS):
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

    APS[(APS == 0).sum(axis=1)==len(APS.columns.values)]=np.nan # some rows are all zero -> set them to nan
    part_legend= np.array(list(map(float, APS.columns.values)))

    # maybe turn below into a function
    aps_agg = pd.DataFrame()
    aps_agg_meta = defaultdict(list)

    #APS scale of dN/dlogDp
    #plt.plot( 1/(np.log10(part_legend[1:]) - np.log10(part_legend[0:-1])) )
    aps_scale = 1/np.mean((np.log10(part_legend[1:]) - np.log10(part_legend[0:-1])))
    print(aps_scale)

    for AGG_WINDOW in AGG_WINDOWS:
        print(AGG_WINDOW)
        agg_str = 'APS_'+str(round(AGG_WINDOW[0]*1000))+'_'+str(round(AGG_WINDOW[1]*1000)) # give in nm to avoid the .
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