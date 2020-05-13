# *** windvectorcoordinates.py ***
# a collection of useful functions to deal with wind vector data and GPS coordiates
# written by Sebastian Landwehr^{1,2} for the ACE-DATA/ASAID project (PI Julia Schmale^{1,2})
# {1} Paul Scherrer Institute, Laboratory of Atmospheric Chemistry, Villigen, Switzerland
# {2} Extreme Environments Research Laboratory,  École Polytechnique Fédérale de Lausanne, School of Architecture, Civil and Environmental Engineering, Lausanne, Switzerland
#
# Copyright 2017-2018 - Swiss Data Science Center (SDSC)
# A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and
# Eidgenössische Technische Hochschule Zürich (ETHZ).
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
        :returns: WDR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
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
        :returns: WDR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
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
        :returns: WDR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
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
        :returns: WDR: relative wind direction [0, 360), where 0 denotes wind blowing against the ship and +90 wind comming from starbroard
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
        :returns: WDR_err: estimated uncertainty in the relative wind direction [degrees]
        
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



# Some ACE wind and track data specific functions ...

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
    """
        Dataset specific function to correctly resample track data [latitude, longitude, HEADING, COG, SOG, velNorth, velEast]
        
        :param df_wind: data fram containing the fields [latitude, longitude, HEADING, COG, SOG, velNorth, velEast]
        :param Nmin: desired time resolution in minutes
        :param interval_center: optional input 'odd' or 'even'
        :param lon_flip_tollerance: tunig parameter to set at which degree of descreptancy the median is used rather than the mean for averaging the longitudes  (suggested 0.0005 for Nmin=1min, and 0.1 for Nmin 5min to 1hour)

        returns: wind_5min: resampled data frame with additional fields [HEADING_DIFF, SOG_DIFF]
               
    """
    
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
        print('interval_center must be "odd" or "even" ...')
        return
        
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

    # now resample the main time series
    wind_5min=wind_5min.resample(str(Nmin)+'T', loffset = loffset).mean() 

    # here we fix dateline issues of averaging longitudes around +/-180 degree
    wind_5min.at[(np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance), 'longitude']=lon_median[np.abs(wind_5min.longitude-lon_median)>lon_flip_tollerance]
    
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
    """
        Dataset specific function to correctly resample wind and track data
        
        :param df_wind: data fram containing the fields: [latitude, longitude, HEADING, COG, SOG, velNorth, velEast, ...
            'u1', 'v1', 'uR1', 'vR1', 'WS1', 'WD1', 'WSR1', 'WDR1', ...
            'u2', 'v2', 'uR2', 'vR2', 'WS2', 'WD2', 'WSR2', 'WDR2']
        :param Nmin: desired time resolution in minutes
        :param interval_center: optional input 'odd' or 'even'
        :param lon_flip_tollerance: tunig parameter to set at which degree of descrptancy the median is used rather than the mean for averaging the longitudes (suggested 0.0005 for Nmin=1min, and 0.1 for Nmin 5min to 1hour)

        returns: wind_5min: resampled data frame with additional fields [HEADING_DIFF, SOG_DIFF, WDR1_DIFF, WDR2_DIFF]
               
    """
    
    wind_5min = df_wind.copy(); # make copy (may not be necessary here)

    # average to Nmin
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



    # now resample the main time series
    wind_5min=wind_5min.resample(str(Nmin)+'T', loffset = loffset).mean() 
    
    # here we fix dateline issues of averaging longitudes around +/-180 degree
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

    # rebuild the angles:
    wind_5min.COG = (90-np.rad2deg(np.arctan2(wind_5min.velNorth,wind_5min.velEast))) % 360 # recompute COG from averaged North/Easte velocities
    wind_5min.HEADING = np.rad2deg(np.arctan2(wind_5min.hdg_sin, wind_5min.hdg_cos)) % 360


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
