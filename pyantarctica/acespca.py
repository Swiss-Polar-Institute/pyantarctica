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
import pandas as pd
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from pathlib import Path as Path

##############################################################################################################

def get_LV_names_20200518():
    # LV names defined for the 20200518 run
    # this list is used to assing LVs with the correct name e.g. in get_LV_list
    # names updates on 27.01.2021 to compare with g-doc
    LV_names_20200518 = [
        "LV names 20200518",  # LV0 to be able to count from 1
        "Climatic zones and large-scale horizontal gradients",  # LV1
        "Meridional cold and warm air advection",  # LV2
        "Wind-driven conditions and sea-spray aerosol",  # LV3
        "Precipitation vs. dry conditions",  # LV4
        "Distance-to-land",  # LV5
        "Drivers of CCN population",  # LV6 Aged secondary aerosol
        "Iron-fertilized biological productivity",  # LV7
        "Iron-limited biological productivity",  # LV8
        "Marginal sea ice zone and snowfall",  # LV9
        "Seasonal signal",  # LV10
        "Surface nutrient concentrations associated with mixing events, climatic, and frontal zones",  # LV11
        "Diel cycle",  # LV12
        "Climatic zones with local high-latitude hotspots",  # LV13
        "Extratropical cyclone activity",  # LV14
        "Bio-aerosols particles",  # LV15
    ]
    return LV_names_20200518


def get_LV_list(RUN_DATE="20200821", DISPLAY_VERSION="Draft"):
    """
        function to create a LV name,sing,display-number list to use in the plotting of the sPCA results
        Knowledge on the runs is hardcoded here!
        
        :params RUN_DATE: Date of the sPCA run "yyyymmdd"
        :params VERSION: display version can be "Draft" or "Final", for VERSION=="Draft" the LVs are adjusted to match the LV run from 20200515
        :returns: LV_list=pd.DataFrame(index=[1..nLV],columns={'LV_name', 'LV_sign', 'LV_display_number'})
    """

    # LV names defined for the 20200821
    LV_names_20200518 = get_LV_names_20200518()

    # set the number of LVs based on the rundate
    if RUN_DATE in ["20200518"]:
        nLV = 15
    elif RUN_DATE in ["20200821"]:
        nLV = 14
    elif RUN_DATE in ["20201104"]: # 720min run compare to 0821
        nLV = 14
    # initiate LV_list and set defaults
    LV_list = pd.DataFrame(
        index=np.arange(1, nLV + 1), columns={"LV_name", "LV_sign", "nLV_display"}
    )
    LV_list["LV_sign"] = "plus"  # no sign change when displaying the LV
    LV_list[["LV_sign"]] = LV_list[["LV_sign"]].astype(str)
    for jLV in np.arange(1, nLV + 1):
        LV_list["nLV_display"][jLV] = int(jLV)
    LV_list[["nLV_display"]] = LV_list[["nLV_display"]].astype(int)

    if RUN_DATE == "20200518":
        for jLV in np.arange(1, nLV + 1):
            LV_list["LV_name"].loc[jLV] = LV_names_20200518[jLV]
            LV_list["LV_sign"].loc[jLV] = "plus"

    elif RUN_DATE == "20200821":

        LV_list["LV_name"].loc[1] = LV_names_20200518[1]
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[1] = "minus"
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[1] = "plus"

        LV_list["LV_name"].loc[2] = LV_names_20200518[6]
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[2] = "minus"
            LV_list["nLV_display"].loc[2] = 6
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[2] = "plus"  # to have Accumulation mode positive

        LV_list["LV_name"].loc[3] = LV_names_20200518[2]
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[3] = "minus"
            LV_list["nLV_display"].loc[3] = 2
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[3] = "plus"

        LV_list["LV_name"].loc[4] = LV_names_20200518[4]  # "Precipitation"
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[4] = "plus"  #
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[4] = "minus"  # to have rain positive

        LV_list["LV_name"].loc[5] = LV_names_20200518[5]  # "Distance to land"
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[5] = "plus"
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[5] = "minus"

        LV_list["LV_name"].loc[6] = LV_names_20200518[7]  # "Iron-fertilized blooms"
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[6] = "minus"
            LV_list["nLV_display"].loc[6] = 7
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[6] = "minus"

        LV_list["LV_name"].loc[7] = LV_names_20200518[10]  # "Seasonal signal"
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[7] = "plus"
            LV_list["nLV_display"].loc[7] = 10

        LV_list["LV_name"].loc[8] = LV_names_20200518[8]  # "Iron-limited biological productivity"
        LV_list["LV_sign"].loc[8] = "plus"

        LV_list["LV_name"].loc[9] = LV_names_20200518[9]  #
        LV_list["LV_sign"].loc[9] = "plus"

        LV_list["LV_name"].loc[10] = LV_names_20200518[12]  # "Diel cycle"
        if DISPLAY_VERSION == "Draft":
            LV_list["nLV_display"].loc[10] = 12
            LV_list["LV_sign"].loc[10] = "plus"
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[10] = "minus"

        LV_list["LV_name"].loc[11] = LV_names_20200518[11]  # "Surface nutrient conc."
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[11] = "plus"
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[11] = "minus"

        LV_list["LV_name"].loc[12] = LV_names_20200518[3]  # "Wind driven conditions and sea spray"
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[12] = "plus"
            LV_list["nLV_display"].loc[12] = 3
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[12] = "minus"

        LV_list["LV_name"].loc[13] = LV_names_20200518[14]
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[13] = "minus"
            LV_list["nLV_display"].loc[13] = 14
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[13] = "minus"  #

        LV_list["LV_name"].loc[14] = LV_names_20200518[13]
        if DISPLAY_VERSION == "Draft":
            LV_list["LV_sign"].loc[14] =  "minus" #"plus" #
            LV_list["nLV_display"].loc[14] = 13
        elif DISPLAY_VERSION == "Final":
            LV_list["LV_sign"].loc[14] =  "plus" #"minus" 
    return LV_list


def get_category_colors(OV_Category):
    # definition of the observed variable categories
    Category_colors = np.unique(OV_Category)
    Category_colors[np.unique(OV_Category) == "Atm. dyn."] = "silver"
    Category_colors[np.unique(OV_Category) == "Atm. hydro."] = "deepskyblue"
    Category_colors[np.unique(OV_Category) == "Atm. chem."] = "magenta"
    Category_colors[np.unique(OV_Category) == "O. dyn."] = "gray"
    Category_colors[np.unique(OV_Category) == "O. hydro."] = "mediumblue"
    Category_colors[np.unique(OV_Category) == "O. biogeochem."] = "darkmagenta"
    Category_colors[np.unique(OV_Category) == "O. microb."] = "green"
    Category_colors[np.unique(OV_Category) == "Topo."] = "saddlebrown"

    return Category_colors


def Mask_LV_Series(weights_lv, timeseries_lv, data, TopFrac=0.5, MinFracOfTopFrac=0.5):
    # weights_lv, timeseries_lv as they come out of SPCA_align_bootstraps()
    # TopFrac=0.5 to use the to 50% larges OV weight medians as reference
    # MinFracOfTopFrac=0.5 to require that 50% of the TopFrac are present

    ### LIST TOP 1-TOP_PC SORTED VARIABLES FOR EACH LV
    med_w = np.median(weights_lv, axis=0)  # median of the 20-bootstrap weights
    num_ov = int(
        np.ceil(TopFrac * np.sum(med_w != 0))
    )  # numer of OVs to count, here based on top most 25%
    inds_var = np.argsort(-np.abs(med_w))[:num_ov]
    if 0:
        # remove cold_warm_mask cause it is never NaN
        inds_var = np.delete(
            inds_var,
            np.where(inds_var == np.where(NAME_plot == "cold_warm_mask-p11")[0][0]),
        )
    n_min_ov = int(
        MinFracOfTopFrac * num_ov
    )  # require to have at least 50% of the selcted top most TopFrac*100%

    blank = add_legs_index(data)["leg"].isna()

    timeseries_lv = pd.DataFrame(timeseries_lv, index=data.index)
    t_s = timeseries_lv.copy()
    t_s.loc[blank, :] = np.nan
    mu = t_s.mean(axis=1)
    sigma = 2 * t_s.std(axis=1)

    mu[(np.sum(~(data.iloc[:, inds_var[:]] == 0.0), axis=1) < n_min_ov)] = np.NaN

    return mu, sigma
