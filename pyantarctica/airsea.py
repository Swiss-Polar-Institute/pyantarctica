# some usefull functions based on air-sea literature
# I have coded these to my best knowledge, but I cannot quarantee for the content to be correct.
# last reviewed by Sebastian Landwehr PSI 20.08.2018

import numpy as np



def coare_u10_ustar (u, input_string='u10', coare_version='coare3.5'):
    # function coare_u10_ustar (u,coare_direction,coare_version) 
    # uses wind speed dependend drag coefficient to iteratively convert between u* and u10n
    #
    # the input is procesed dependend on the input_string
    # for input_string=='u10': coare_u10_ustar converts u10(neutral)->u*
    # for input_string=='ustar': coare_u10_ustar converts u*->u10(neutral)
    #
    # coare_version defines which drag coefficient is used for the conversion
    # coare_version='coare3.5' use wind speed dependend charnock coefficient coare version 3.5 Edson et al. 2013
    # coare_version='coare3.0' use wind speed dependend charnock coefficient coare version 3.0 Fairall et al. 2003
    # for citing this code please refere to:  
    # https://www.atmos-chem-phys.net/18/4297/2018/ equation (4),(5), and (6)
    # Sebastian Landwehr, PSI 2018


    
    import numpy as np
    if input_string == 'ustar':
        ustar = u;
        u10n = 30*ustar; # first guess
    elif input_string == 'u10':
        u10n = u;
        ustar = u10n/30;
    else:
        print('unexpected "input_string"! please use "u10" or "ustar"')

        
    t=20; # air temperature [C]
    grav = 9.82; # const of gravitation
    vkarman = 0.4;
    gamma = 0.11; # roughness Reynolds number
    charnock = 0.011;
    visa=1.326e-5*(1+6.542e-3*t+8.301e-6*t*t-4.84e-9*t*t*t); # viscosity of air


    for jj in [1, 2, 3]:
        if coare_version == 'coare3.5':
            charnock=0.0017*u10n-0.005; # note EDSON2013 gives this as 0.017*U10-0.005 BUT from plot it must be 0.0017!!!
            charnock[u10n>19.4]=0.028; # charnock(19.4)~0.028
        elif coare_version == 'coare3.0':
            charnock=0.00225+0.007/8*u10n; # Fairall2003 a=0.011@u=10 and a=0.018@u=18
            charnock[u10n>18]=0.018; 
            charnock[u10n<10]=0.011; 
        else:
            print('unexpected "coare_version"! please use "coare3.5" or "coare3.0"')


        z0 = gamma*(visa/ustar)+charnock*ustar*ustar/grav;
        sqrt_C_D = (vkarman/np.log(10/z0));

        if input_string == 'ustar':
            u10n=(ustar/sqrt_C_D);
        elif input_string == 'u10':
            ustar=(u10n*sqrt_C_D);

    if input_string == 'ustar':
        u=u10n
    elif input_string == 'u10':
        u=ustar
        
    return u