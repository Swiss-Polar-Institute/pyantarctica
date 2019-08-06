import datetime
import numpy as np

# collection of useful functions to deal with wind vector data and GPS coordiates
def ang180(x):
    return ((x+180)%360)-180 # map any number into -180, 180 keeping the order the same

def UVrel2WSRWDR(Urel,Vrel,HEADING):
    # Urel,Vrel in earth coordinate system!!!
    WSR = np.sqrt(np.square(Urel)+np.square(Vrel))
    WDR = (270-HEADING-np.rad2deg(np.arctan2(Vrel,Urel) ) )%360 # 
    return WSR, WDR

def UVtrue2UVrel(U,V,velEast,velNorth):
    # from U,V, HEADING and east north velocity calculate relative wind speed and direction
    Urel = U-velEast
    Vrel = V-velNorth
    return Urel, Vrel

def UVtrue2WSRWDR(U,V,HEADING,velEast,velNorth):
    Urel, Vrel = UVtrue2UVrel(U,V,velEast,velNorth)
    WSR, WDR = UVrel2WSRWDR(Urel,Vrel,HEADING)
    return WSR, WDR

def WSRWDR2UVtrue(WSR,WDR,HEADING,velEast,velNorth):
    uA=WSR*np.cos(np.deg2rad(270-HEADING-WDR))
    vA=WSR*np.sin(np.deg2rad(270-HEADING-WDR))
    U = uA+velEast
    V = vA+velNorth
    return U, V
    
def UVtrue2WSWD(U,V):
    WS = np.sqrt(np.square(U)+np.square(V))
    WD = (270 - np.rad2deg(np.arctan2(V, U)) )% 360
    return WS, WD

def WSWD2UVtrue(WS,WD):
    U=WS*np.cos(np.deg2rad(270-WD))
    V=WS*np.sin(np.deg2rad(270-WD))
    return U, V


def WSWD2WSRWDR(WS,WD,HEADING,velEast,velNorth):
    U, V = WSWD2UVtrue(WS,WD)
    WSR, WDR = UVtrue2WSRWDR(U,V,HEADING,velEast,velNorth)
    return WSR, WDR



# ACE specific functions ...
def dirdiff(HEADING,Nmin,loffset):
    HEADING_MAX=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING = (HEADING-180)%360
    HEADING_MAX_=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN_=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING_DIFF = np.min([(HEADING_MAX-HEADING_MIN), (HEADING_MAX_-HEADING_MIN_)], axis=0)
    return HEADING_DIFF


def resample_wind_data(df_wind, Nmin=5,interval_center='odd', lon_flip_tollerance=0.0005):
    

    #lon_flip_tollerance = 0.0005 # for 1min to 5min a value well aboved the normal difference of mean and median longitudes
    #lon_flip_tollerance = 0.1 # for 1min to 1hour


    wind_5min = df_wind.copy(); # make copy (may not be necessary here)

    # average to 5min
    if interval_center == 'odd':
        loffset = datetime.timedelta(minutes=0.5*Nmin)
    # adjust the loffset to get the timestamp into the center of the interval
    # this leads to mean bins at MM:30
    elif interval_center == 'even':
        loffset = datetime.timedelta(minutes=0*Nmin)
        wind_5min.index=wind_5min.index+datetime.timedelta(minutes=.5*Nmin);
    else:
        print('interval_center must be odd or even')

    #wind_5min_STD=df_wind.resample(str(Nmin)+'T', loffset = loffset).std()
    # calculate max heading difference in degree (very linear with HEDING_STD)
    HEADING = wind_5min.HEADING.copy();
    HEADING_MAX=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING = (HEADING-180)%360
    HEADING_MAX_=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN_=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING_DIFF = np.min([(HEADING_MAX-HEADING_MIN), (HEADING_MAX_-HEADING_MIN_)], axis=0)

    WDR1_DIFF = dirdiff(wind_5min.WDR1,Nmin,loffset)
    WDR2_DIFF = dirdiff(wind_5min.WDR2,Nmin,loffset)
    
    SOG = wind_5min.SOG.copy();
    SOG_MAX=SOG.resample(str(Nmin)+'T', loffset = loffset).max()
    SOG_MIN=SOG.resample(str(Nmin)+'T', loffset = loffset).min()

    #lon_median = wind_5min.longitude.copy();
    lon_median=wind_5min['longitude'].resample(str(Nmin)+'T', loffset = loffset).median() # median of longitudes

    # here we fix dateline issues of averaging longitudes around +/-180 degree



    # now resample the main time series
    wind_5min=wind_5min.resample(str(Nmin)+'T', loffset = loffset).mean() 


   #wind_5min['longitude'][np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance]=lon_median[np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance]
    wind_5min.at[(np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance), 'longitude']=lon_median[np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance]

    #wind_5min = wind_5min.assign(HEADING_DIFF=HEADING_DIFF)
    #wind_5min = wind_5min.assign(SOG_DIFF=(SOG_MAX-SOG_MIN))
    wind_5min.HEADING_DIFF=HEADING_DIFF
    wind_5min.SOG_DIFF=(SOG_MAX-SOG_MIN)
    
    wind_5min = wind_5min.assign(WDR1_DIFF=WDR1_DIFF)
    wind_5min = wind_5min.assign(WDR2_DIFF=WDR2_DIFF)


    #, 'u1', 'v1', 'uR1', 'vR1'    
    #wind_5min = wind_5min.merge(wind_5min_STD[['SOG', 'velNorth', 'velEast', 'HEADING', 'hdg_sin', 'hdg_cos']], left_on='timest_', right_on='timest_', how='inner', suffixes=('', '_STD'))
    # rebuild the angles: ! chech on dirs
    wind_5min.COG = (90-np.rad2deg(np.arctan2(wind_5min.velNorth,wind_5min.velEast))) % 360 # recompute COG from averaged North/Easte velocities
    wind_5min.HEADING = np.rad2deg(np.arctan2(wind_5min.hdg_sin, wind_5min.hdg_cos)) % 360
    #wind_5min.HEADING_STD = np.rad2deg(np.arctan2(wind_5min.hdg_sin_STD, wind_5min.hdg_cos_STD))
    #wind_5min.HEADING_STD = np.sqrt( np.square(wind_5min.hdg_sin_STD) + np.square(wind_5min.hdg_cos_STD) )

    wind_5min.WD1 = (270 - np.rad2deg(np.arctan2(wind_5min.v1, wind_5min.u1)) )% 360
    wind_5min.WDR1 = (180 - np.rad2deg(np.arctan2(wind_5min.vR1, wind_5min.uR1)) )% 360
    wind_5min.WD2 = (270 - np.rad2deg(np.arctan2(wind_5min.v2, wind_5min.u2)) )% 360
    wind_5min.WDR2 = (180 - np.rad2deg(np.arctan2(wind_5min.vR2, wind_5min.uR2)) )% 360


    # recalcualte the speeds as vector average
    wind_5min.SOG = np.sqrt( np.square(wind_5min.velNorth) + np.square(wind_5min.velEast) )

    wind_5min.WS1 = np.sqrt( np.square(wind_5min.v1) + np.square(wind_5min.u1) )
    wind_5min.WS2 = np.sqrt( np.square(wind_5min.v2) + np.square(wind_5min.u2) )
    wind_5min.WSR1 = np.sqrt( np.square(wind_5min.vR1) + np.square(wind_5min.uR1) )
    wind_5min.WSR2 = np.sqrt( np.square(wind_5min.vR2) + np.square(wind_5min.uR2) )

    return wind_5min