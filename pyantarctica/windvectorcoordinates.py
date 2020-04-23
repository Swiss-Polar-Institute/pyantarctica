# collection of useful functions to deal with wind vector data and GPS coordiates
# last reviewed by Sebastian Landwehr PSI 27.03.2020

import datetime
import numpy as np

def ang180(x):
    """
        Function to mapp angular data over the interval [-180 +180)

        :param x: data series
        :returns: the data series mapped into [-180 +180)
    """
    return ((x+180)%360)-180 # map any number into -180, 180 keeping the order the same

def URVRship2WSRWDR(Urel,Vrel):
    """
        Function to calculate relative wind speed and relative wind direction [0, 360) from the relaive wind vector [U_s,V_s] in ships reference frame (right hand coordinate system)

        :param Urel: relative wind speed along ships main axis
        :param Vrel: relative wind speed perpendicular to the ships main axis

        :returns: WSR: relative wind speed (absolute value)
        :returns: WSR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
    """
    # this is for Urel,Vrel in ship coordinate system
    WSR = np.sqrt(np.square(Urel)+np.square(Vrel))
    WDR = (180-np.rad2deg(np.arctan2(Vrel,Urel) ) )%360 # 
    return WSR, WDR

def UVrel2WSRWDR(Urel,Vrel,HEADING):
    """
        Function to calculate relative wind speed and relative wind direction [0, 360) from the relative wind vector [U_Earth,V_Earth] in Earth reference frame (right hand coordinate system)

        :param Urel: relative wind speed in East direction
        :param Vrel: relative wind speed in Northward direction
        :param HEADING: ships heading [0, 360) clockwise from North

        :returns: WSR: relative wind speed (absolute value)
        :returns: WSR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
    """
    # this is for Urel,Vrel in earth coordinate system, the way they come out of UVtrue2UVrel!!!
    WSR = np.sqrt(np.square(Urel)+np.square(Vrel))
    WDR = (270-HEADING-np.rad2deg(np.arctan2(Vrel,Urel) ) )%360 # 
    return WSR, WDR

def UVtrue2UVrel(U,V,velEast,velNorth):
    """
        Function to calculate the relative wind vector from the vector combination of the true wind vector [U,V] and the ships velocity [V_Eeast,V_North]. All in Earth reference frame (right hand coordinate system)

        :param U: true wind speed in East direction
        :param V: true wind speed in North direction
        :param velEast: ships velocity in East direction
        :param velNorth: ships velocity in North direction

        :returns: Urel: relative wind speed in East direction
        :returns: Vrel: relative wind speed in Northward direction
    """
    Urel = U-velEast
    Vrel = V-velNorth
    return Urel, Vrel

def UVtrue2WSRWDR(U,V,HEADING,velEast,velNorth):
    """
        Function to calculate the relative wind speed and relative wind direction [0, 360) from the vector combination of the true wind vector [U,V] and the ships velocity [V_Eeast,V_North] (all in Earth reference frame), followed by a rotation by the ships HEADING to end up in the ships reference frame
        (This is the invers of the True wind speed correction)

        :param U: true wind speed in East direction
        :param V: true wind speed in North direction
        :param HEADING: ships heading [0, 360) clockwise from North
        :param velEast: ships velocity in East direction
        :param velNorth: ships velocity in North direction
        
        :returns: WSR: relative wind speed (absolute value)
        :returns: WSR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
    """        
    Urel, Vrel = UVtrue2UVrel(U,V,velEast,velNorth)
    WSR, WDR = UVrel2WSRWDR(Urel,Vrel,HEADING)
    return WSR, WDR

def WSRWDR2UVtrue(WSR,WDR,HEADING,velEast,velNorth):
    """
        True wind speed following Smith et al 1999
        Function to calculate the true wind vector from the relative wind speed and relative wind direction [0, 360) the ships heading and the ships velocity [V_Eeast,V_North]

        :param WSR: relative wind speed (absolute value)
        :param WDR: relative wind direction [0, 360)
        :param HEADING: ships heading [0, 360) clockwise from North
        :param velEast: ships velocity in East direction
        :param velNorth: ships velocity in North direction
        
        :returns: U: true wind speed in East direction
        :returns: V: true wind speed in North direction
    """    
    uA=WSR*np.cos(np.deg2rad(270-HEADING-WDR))
    vA=WSR*np.sin(np.deg2rad(270-HEADING-WDR))
    U = uA+velEast
    V = vA+velNorth
    return U, V
    
def UVtrue2WSWD(U,V):
    """
        Function to calculate true wind speed and true wind direction [0, 360) from the true wind vector [U_Earth,V_Eearth] in Earth reference frame (right hand coordinate system)

        :param U: true wind speed in East direction
        :param V: true wind speed in North direction

        :returns: WS: true wind speed (absolute value)
        :returns: WD: true wind direction [0, 360), where 0 denotes wind blowing from North +90 wind comming from East
    """
    WS = np.sqrt(np.square(U)+np.square(V))
    WD = (270 - np.rad2deg(np.arctan2(V, U)) )% 360
    return WS, WD

def WSWD2UVtrue(WS,WD):
    """
        Function to calculate true wind vecotr from the true wind speed and true wind direction [0, 360)
        
        :param WS: true wind speed (absolute value)
        :param WD: true wind direction [0, 360), where 0 denotes wind blowing from North +90 wind comming from East
        
        :returns: U: true wind speed in East direction
        :returns: V: true wind speed in North direction        
    """
    U=WS*np.cos(np.deg2rad(270-WD))
    V=WS*np.sin(np.deg2rad(270-WD))
    return U, V


def WSWD2WSRWDR(WS,WD,HEADING,velEast,velNorth):
    """
        Function to calculate the relative wind speed and relative wind direction [0, 360) from true wind speed, true wind direction, ships heading and velocity vector
        
        :param WS: true wind speed (absolute value)
        :param WD: true wind direction [0, 360), where 0 denotes wind blowing from North +90 wind comming from East
        :param HEADING: ships heading [0, 360) clockwise from North
        :param velEast: ships velocity in East direction
        :param velNorth: ships velocity in North direction
          
        :returns: WSR: relative wind speed (absolute value)
        :returns: WSR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
    """    
    U, V = WSWD2UVtrue(WS,WD)
    WSR, WDR = UVtrue2WSRWDR(U,V,HEADING,velEast,velNorth)
    return WSR, WDR


# function to estimate how sensitive predicted relative wind speed and direction are on biased TRUE WIND input
def WSRWDR_uncertainy(WSPD,WDIR,HEADING,velEast,velNorth,a_WSPD=1.1,d_WDIR=10):
    """
        Function to calculate the uncertainty of relative wind speed and relative wind direction that have been estimated from a true wind speed, true wind direction, ships heading, and velocity vector, where the true wind speed and direction are uncertain by a factor/angle
        
        :param WSPD: true wind speed (absolute value)
        :param WDIR: true wind direction [0, 360), where 0 denotes wind blowing from North +90 wind comming from East
        :param HEADING: ships heading [0, 360) clockwise from North
        :param velEast: ships velocity in East direction
        :param velNorth: ships velocity in North direction
        :param a_WSPD: specified uncertainty in the true wind speed (use 1.1 to denote 10% uncertainty)
        :param d_WDIR: specified uncertainty in the true wind direction [degrees]
          
        :returns: WSR_err: estimated uncertainty in the relative wind speed [m/s]
        :returns: WSR_err: estimated uncertainty in the relative wind direction [degrees]
        
        See Appendix section B: From errors in the reference wind vector to errors in the expected relative wind speed and direction
        in Landwehr et al. (2020) ``Using global reanalysis data to quantify and correct airflow distortion bias in shipborne wind speed measurements''
        
    """    
    WSR, WDR = WSWD2WSRWDR(WSPD,WDIR,HEADING,velEast,velNorth) # basline
    WSR_aup, WDR_aup = WSWD2WSRWDR(WSPD*a_WSPD,WDIR,HEADING,velEast,velNorth) # vary WSPD up by factor
    WSR_alo, WDR_alo = WSWD2WSRWDR(WSPD/a_WSPD,WDIR,HEADING,velEast,velNorth)# vary WSPD down by factor
    
    d_WSPD = np.max([WSPD*(a_WSPD-1), np.ones_like(WSPD)], axis=0) # error x% or 1m/s
    
    #d_WSPD[(d_WSPD>WSPD)]=WSPD[(d_WSPD>WSPD)] # max error of 100%
    WSR_aup, WDR_aup = WSWD2WSRWDR(WSPD+d_WSPD,WDIR,HEADING,velEast,velNorth) # vary WSPD up by factor
    WSR_alo, WDR_alo = WSWD2WSRWDR(WSPD-d_WSPD,WDIR,HEADING,velEast,velNorth)# vary WSPD down by factor
    
    WSR_dup, WDR_dup = WSWD2WSRWDR(WSPD,WDIR+d_WDIR,HEADING,velEast,velNorth) # vary WDIR up by degree
    WSR_dlo, WDR_dlo = WSWD2WSRWDR(WSPD,WDIR-d_WDIR,HEADING,velEast,velNorth)#  vary WDIR up by degree

    # estimate the uncertainty in WSR by looking for the maximal deviation caused by the variant input
    WSR_err = np.max([np.abs(WSR_aup-WSR),
                      np.abs(WSR_alo-WSR),
                      np.abs(WSR_dup-WSR),
                      np.abs(WSR_dlo-WSR),
                     ], axis=0)

    WDR_err = np.max([np.abs(ang180(WDR_aup-WDR)),
                      np.abs(ang180(WDR_alo-WDR)),
                      np.abs(ang180(WDR_dup-WDR)),
                      np.abs(ang180(WDR_dlo-WDR)),
                     ], axis=0)

    return WSR_err, WDR_err



# ACE specific functions ...
def dirdiff(HEADING,Nmin,loffset):
    """
        Function to calculate the maximum difference of a [0, 360) direction during specified time averging intervals
        
        returns: HEADING_DIFF: time series of the maximal difference between the direction estimates during the specified averaging interval
               
    """
    HEADING_MAX=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING = (HEADING-180)%360
    HEADING_MAX_=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN_=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING_DIFF = np.min([(HEADING_MAX-HEADING_MIN), (HEADING_MAX_-HEADING_MIN_)], axis=0)
    return HEADING_DIFF

def resample_track_data(df_wind, Nmin=5,interval_center='odd', lon_flip_tollerance=0.0005):
    # suggested input for lon_flip_tollerance:
    #lon_flip_tollerance = 0.0005 # for Nmin=1min a value well aboved the normal difference of mean and median longitudes
    #lon_flip_tollerance = 0.1 # for Nmin 5min to 1hour

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
        
    # calculate max heading difference in degree (very linear with HEDING_STD)
    HEADING_DIFF = dirdiff(wind_5min.HEADING.copy(),Nmin,loffset)

    SOG = wind_5min.SOG.copy();
    SOG_MAX=SOG.resample(str(Nmin)+'T', loffset = loffset).max()
    SOG_MIN=SOG.resample(str(Nmin)+'T', loffset = loffset).min()
    
    # caclulate heading vecotor resampled
    hdg_cos=( np.cos(np.deg2rad(wind_5min.HEADING)) ).resample(str(Nmin)+'T', loffset = loffset).mean() 
    hdg_sin=( np.sin(np.deg2rad(wind_5min.HEADING)) ).resample(str(Nmin)+'T', loffset = loffset).mean() 

    
    #lon_median = wind_5min.longitude.copy();
    lon_median=wind_5min['longitude'].resample(str(Nmin)+'T', loffset = loffset).median() # median of longitudes
    # here we fix dateline issues of averaging longitudes around +/-180 degree

    # now resample the main time series
    wind_5min=wind_5min.resample(str(Nmin)+'T', loffset = loffset).mean() 

   #wind_5min['longitude'][np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance]=lon_median[np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance]
    wind_5min.at[(np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance), 'longitude']=lon_median[np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance]
    
    #wind_5min = wind_5min.assign(HEADING_DIFF=HEADING_DIFF)
    #wind_5min = wind_5min.assign(SOG_DIFF=(SOG_MAX-SOG_MIN))
    if 'HEADING_DIFF' in wind_5min.columns:
        wind_5min['HEADING_DIFF']=HEADING_DIFF
    else:
        wind_5min = wind_5min.assign( HEADING_DIFF = HEADING_DIFF )
    if 'SOG_DIFF' in wind_5min.columns:
        wind_5min['SOG_DIFF']=(SOG_MAX-SOG_MIN)
    else:
        wind_5min = wind_5min.assign( SOG_DIFF=(SOG_MAX-SOG_MIN) )

    wind_5min.COG = (90-np.rad2deg(np.arctan2(wind_5min.velNorth,wind_5min.velEast))) % 360 # recompute COG from averaged North/Easte velocities
    wind_5min.HEADING = np.rad2deg(np.arctan2(hdg_sin, hdg_cos)) % 360 # recompute HEADING from average components
    # recalcualte the speeds as vector average
    wind_5min.SOG = np.sqrt( np.square(wind_5min.velNorth) + np.square(wind_5min.velEast) )

    return wind_5min


def resample_wind_data(df_wind, Nmin=5,interval_center='odd', lon_flip_tollerance=0.0005):
    
    # suggested input for lon_flip_tollerance:
    #lon_flip_tollerance = 0.0005 # for Nmin=1min a value well aboved the normal difference of mean and median longitudes
    #lon_flip_tollerance = 0.1 # for Nmin 5min to 1hour


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
 
    if 'HEADING_DIFF' in wind_5min.columns:
        wind_5min['HEADING_DIFF']=HEADING_DIFF
    else:
        wind_5min = wind_5min.assign( HEADING_DIFF = HEADING_DIFF )
        
    if 'SOG_DIFF' in wind_5min.columns:
        wind_5min['SOG_DIFF']=(SOG_MAX-SOG_MIN)
    else:
        wind_5min = wind_5min.assign( SOG_DIFF=(SOG_MAX-SOG_MIN) )
    
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


def wind_merge_gps_afc_option(gps_file,wind_file,merge_at_nseconds,afc_loess_files=[]):
    from scipy.interpolate import interp1d # for afc correction

    df_gps = pd.read_csv(gps_file)
    df_gps = df_gps.rename(index=str, columns={"Unnamed: 0": "timest_"})
    df_gps = df_gps.set_index( (pd.to_datetime(df_gps.timest_, format="%Y-%m-%d %H:%M:%S"))  ) # assing the time stamp
    df_gps = df_gps.drop(columns=['timest_'])
    df_gps.index.name = 'timest_'

    df_wind = pd.read_csv(wind_file)
    df_wind = df_wind.rename(index=str, columns={"Unnamed: 0": "timest_"})
    df_wind = df_wind.set_index( (pd.to_datetime(df_wind.timest_, format="%Y-%m-%d %H:%M:%S"))  ) # assing the time stamp
    df_wind = df_wind.drop(columns=['timest_'])
    df_wind.index.name = 'timest_'
    
    df_wind.at[ ((df_wind.index>pd.to_datetime("2017-04-05 06:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC'))&(df_wind.index<pd.to_datetime("2017-04-05 12:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC')))==1, 'WSR2'] = np.NaN
    df_wind.at[ ((df_wind.index>pd.to_datetime("2017-04-05 06:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC'))&(df_wind.index<pd.to_datetime("2017-04-05 12:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC')))==1, 'WDR2'] = np.NaN

    if len(afc_loess_files)>0: # if afc data provided
        # interpolate onto wind series and calculate afc wind speed and direction

        afc_loess1 = pd.read_csv(afc_loess_files+'1.csv')
        afc_loess2 = pd.read_csv(afc_loess_files+'2.csv')
        
        afc_loess1=afc_expand(afc_loess1)
        afc_loess2=afc_expand(afc_loess2)

        # interpolate bias onto time series
        r1=((df_wind.WDR1)+180)%360-180
        fA1 = interp1d(afc_loess1.R, afc_loess1.A, kind='linear')
        a1 = fA1(r1)
        fD1 = interp1d(afc_loess1.R, afc_loess1.D, kind='linear')
        d1 = fD1(r1)
        # interpolate bias onto time series
        r2=((df_wind.WDR2)+180)%360-180
        fA2 = interp1d(afc_loess2.R, afc_loess2.A, kind='linear')
        a2 = fA2(r2)
        fD2 = interp1d(afc_loess2.R, afc_loess2.D, kind='linear')
        d2 = fD2(r2)

        # WSR = WSR/a
        # WDR =  WDR-d
        #s1
        df_wind.WSR1 = np.true_divide(df_wind.WSR1,a1)
        df_wind.WDR1 = (df_wind.WDR1-d1)
        #s2
        df_wind.WSR2 = np.true_divide(df_wind.WSR2,a2)
        df_wind.WDR2 = (df_wind.WDR2-d2)

    # calculate uR,vR and u,v from SR, DR and WS,WR
    # uR are positive for wind along ships main axis
    # vR are positive for wind in port direction
    # u are positive for wind blowing East
    # v are positive for wind blowing North
    df_wind = df_wind.assign(uR1=df_wind.WSR1*np.cos(np.deg2rad(180-df_wind.WDR1)))
    df_wind = df_wind.assign(vR1=df_wind.WSR1*np.sin(np.deg2rad(180-df_wind.WDR1)))
    df_wind = df_wind.assign(u1=df_wind.WS1*np.cos(np.deg2rad(270-df_wind.WD1)))
    df_wind = df_wind.assign(v1=df_wind.WS1*np.sin(np.deg2rad(270-df_wind.WD1)))
    #
    df_wind = df_wind.assign(uR2=df_wind.WSR2*np.cos(np.deg2rad(180-df_wind.WDR2)))
    df_wind = df_wind.assign(vR2=df_wind.WSR2*np.sin(np.deg2rad(180-df_wind.WDR2)))
    df_wind = df_wind.assign(u2=df_wind.WS2*np.cos(np.deg2rad(270-df_wind.WD2)))
    df_wind = df_wind.assign(v2=df_wind.WS2*np.sin(np.deg2rad(270-df_wind.WD2)))

    # add heading sin cos for proper averaging
    df_wind = df_wind.assign(hdg_cos=np.cos(np.deg2rad(df_wind.HEADING)))
    df_wind = df_wind.assign(hdg_sin=np.sin(np.deg2rad(df_wind.HEADING)))

    # assing apparent wind in eart,north coordinates
    df_wind = df_wind.assign(uA1=df_wind.WSR1*np.cos(np.deg2rad(270-df_wind.HEADING-df_wind.WDR1)))
    df_wind = df_wind.assign(vA1=df_wind.WSR1*np.sin(np.deg2rad(270-df_wind.HEADING-df_wind.WDR1)))
    df_wind = df_wind.assign(uA2=df_wind.WSR2*np.cos(np.deg2rad(270-df_wind.HEADING-df_wind.WDR2)))
    df_wind = df_wind.assign(vA2=df_wind.WSR2*np.sin(np.deg2rad(270-df_wind.HEADING-df_wind.WDR2)))



    nSample_min = 6 # out of 20 possible (usually 12+/2)

    nWSR1=df_wind.WSR1.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*0.5))).count() # calculate 6 second average
    nWSR2=df_wind.WSR2.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*0.5))).count() # calculate 6 second average
    #plt.plot(nWSR1,'.')
    # strangely some bins contain more than nSec/3 wind samples!!!

    print( 'WSR1 removing ' + str( np.sum((nWSR1>0) & (nWSR1<nSample_min)) ) + ' of '+str(np.sum(nWSR1>0)) )
    print( 'WSR2 removing ' + str( np.sum((nWSR2>0) & (nWSR2<nSample_min)) ) + ' of '+str(np.sum(nWSR1>0)) )

    if 1: # compute HEADING DIFF accross 1min interval
        HEADING = df_wind.HEADING.copy();
        HEADING_MAX=HEADING.resample(str(merge_at_nseconds)+'S').max()
        HEADING_MIN=HEADING.resample(str(merge_at_nseconds)+'S').min()
        HEADING = (HEADING-180)%360
        HEADING_MAX_=HEADING.resample(str(merge_at_nseconds)+'S').max()
        HEADING_MIN_=HEADING.resample(str(merge_at_nseconds)+'S').min()
        HEADING_DIFF = np.min([(HEADING_MAX-HEADING_MIN), (HEADING_MAX_-HEADING_MIN_)], axis=0)



    df_wind=df_wind.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*0.5))).mean() # calculate 6 second average

    if 1:
        df_wind = df_wind.assign(HEADING_DIFF=HEADING_DIFF)

        df_wind = df_wind.assign(nWSR1=nWSR1)
        df_wind = df_wind.assign(nWSR2=nWSR2)

    # set data with too few readings to NaN
    df_wind.at[nWSR1<nSample_min, 'uR1'] = np.nan
    df_wind.at[nWSR1<nSample_min, 'vR1'] = np.nan
    df_wind.at[nWSR1<nSample_min, 'uA1'] = np.nan
    df_wind.at[nWSR1<nSample_min, 'vA1'] = np.nan
    df_wind.at[nWSR2<nSample_min, 'uR2'] = np.nan
    df_wind.at[nWSR2<nSample_min, 'vR2'] = np.nan
    df_wind.at[nWSR2<nSample_min, 'uA2'] = np.nan
    df_wind.at[nWSR2<nSample_min, 'vA2'] = np.nan

    # rebuild the angles:
    df_wind.HEADING = np.rad2deg(np.arctan2(df_wind.hdg_sin, df_wind.hdg_cos)) % 360
    df_wind.WD1 = (270 - np.rad2deg(np.arctan2(df_wind.v1, df_wind.u1)) )% 360
    df_wind.WDR1 = (180 - np.rad2deg(np.arctan2(df_wind.vR1, df_wind.uR1)) )% 360
    df_wind.WD2 = (270 - np.rad2deg(np.arctan2(df_wind.v2, df_wind.u2)) )% 360
    df_wind.WDR2 = (180 - np.rad2deg(np.arctan2(df_wind.vR2, df_wind.uR2)) )% 360
    # recalcualte the speeds as vector average
    df_wind.WS1 = np.sqrt( np.square(df_wind.v1) + np.square(df_wind.u1) )
    df_wind.WS2 = np.sqrt( np.square(df_wind.v2) + np.square(df_wind.u2) )
    df_wind.WSR1 = np.sqrt( np.square(df_wind.vR1) + np.square(df_wind.uR1) )
    df_wind.WSR2 = np.sqrt( np.square(df_wind.vR2) + np.square(df_wind.uR2) )

    # merge with GPS on 30sec basis
    df_wind = df_wind.merge(df_gps, left_on='timest_', right_on='timest_', how='inner')

    df_wind.COG = (90-np.rad2deg(np.arctan2(df_wind.velNorth,df_wind.velEast))) % 360 # recompute COG from averaged North/Easte velocities


    # True wind correction using GPS data

    # use average apparent wind for motion correction
    df_wind.u1 = df_wind.uA1 + df_wind.velEast
    df_wind.v1 = df_wind.vA1 + df_wind.velNorth
    df_wind.u2 = df_wind.uA2 + df_wind.velEast
    df_wind.v2 = df_wind.vA2 + df_wind.velNorth

    # rebuild the angles:
    df_wind.WD1 = (270 - np.rad2deg(np.arctan2(df_wind.v1, df_wind.u1)) )% 360
    df_wind.WD2 = (270 - np.rad2deg(np.arctan2(df_wind.v2, df_wind.u2)) )% 360
    # recalcualte the speeds as vector average
    df_wind.WS1 = np.sqrt( np.square(df_wind.v1) + np.square(df_wind.u1) )
    df_wind.WS2 = np.sqrt( np.square(df_wind.v2) + np.square(df_wind.u2) )


    # AT THIS POIN IT MAKES SENSE TO MAKE A BREAK AND SAVE df_wind 1min merged time series
    #df_wind.to_csv(data_intermediate + 'ship_data\\' + 'merged_afc_wind_gps_filtered.csv') # NAME TO BE DEFINED

    return df_wind