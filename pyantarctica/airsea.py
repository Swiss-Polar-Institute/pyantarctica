# some usefull functions based on air-sea literature
# I have coded these to my best knowledge, but I cannot quarantee for the content to be correct.
# last reviewed by Sebastian Landwehr PSI 20.08.2018

import pyantarctica.constants as constants



def PSIu(zeta, option='default'):
    #stability correction function for modifying the logarithmic wind speed profiles based on atmospheric stability
    # use e.g. for: u(z)=u*/k[log(z/z0)-PSIu(z/L)]
    #
    # PSIu is integral of the semiempirical function PHIu
    # PSIu(z/L)=INT_z0^z[1-PHI_u(z/L)]d(z/L)/(z/L)
    # several forms of PHIu and PSIu are published and will be added as options
    # default = 'Dyer_Hicks_1970'
    
    import numpy as np
    # zeta=z/L or is it -z/L ???
    # with L = -u*^3/vkarman/(g<wT>/T+0.61g<wq>)
    #x=np.sqrt(np.sqrt(1-15*zeta)); #sqrt(sqrt) instead of ^.25
    if type(zeta) != np.ndarray:
        zeta = np.array([zeta])
        
    if option == 'default': # or Dyer_Hicks_1970
        # Dyer and Hicks 1970       
        x=zeta*0 # avoid warings
        x[zeta<0]=np.sqrt(np.sqrt(1-15*zeta[zeta<0])); #sqrt(sqrt) instead of ^.25
        psi=2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x)+2*np.arctan(1); 
        psi[zeta>=0]=-5*zeta[zeta>=0];
    elif option == 'Fairall_1996':
        print('todo')
        psi = []
    else:
        print('unexpected option: please use "default"')
        psi = []
            
    return psi


def coare_u2ustar (u, input_string='u2ustar', coare_version='coare3.5', TairC=20, z=10, zeta=0): 
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

    z0 = 1e-4 # default roughness length (could calculate this using first guess charnock and ustar)
    
    if type(u) != np.ndarray:
        u = np.array([u])
    
    import numpy as np
    if input_string == 'ustar2u':
        ustar = u;
        u10n = 30*ustar; # first guess
    elif input_string == 'u2ustar':
        u10n = u*np.log(10/z0)/np.log(z/z0); # first guess u10n for calculating initial charnock
        ustar = u10n/30;
    else:
        print('unexpected "input_string"! please use "u2ustar" or "ustar2u"')

        
    t=TairC; # air temperature [C]
    grav = constants.g; # const of gravitation
    vkarman = constants.vanKarman; # van Karman constant
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
        sqrt_C_D = (vkarman/np.log(z/z0));
        sqrt_C_D = (vkarman/(np.log(z/z0)-PSIu(zeta))); # when adding stability use this equation ...
        sqrt_C_D_10 = (vkarman/np.log(10/z0)); # 10m neutral drag coefficient

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
        
    return u

def coare_u10_ustar (u, input_string='u10', coare_version='coare3.5', TairC=20):
    #TO BE REMOVED
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

        
    t=TairC; # air temperature [C]
    grav = 9.82; # const of gravitation
    vkarman = 0.4; # van Karman constant
    gamma = 0.11; # roughness Reynolds number
    charnock = 0.011; # first guess charnock parameter
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