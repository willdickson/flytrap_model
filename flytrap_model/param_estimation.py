"""
fly_trap_model.py

Provides functions for estimating the coefficients in the linear fly trap model

o[n+1] = o[n] - foh*o[n] + a[n]
h[n+1] = h[n] - fhv*h[n] + fvh*v[n] + foh*o[n]
v[n+1] = v[n] + fhv*h[n] - fvh*v[n] 

where

a[n] = # flies arriving on the trap at time step n 
o[n] = # flies on the trap at time step n
h[n] = # flies in the trap, which are hidden, at time step n
v[n] = # flies in the trap, which are visible, at time step n 

foh = fraction of flies transitioning from o[n] to h[n] during a time step
fhv = fraction of flies transitioning from h[n] to v[n] during a time step
fvh = fraction of flies transitioning from v[n] to h[n] during a time step


"""
from __future__ import print_function
import numpy 
import scipy.signal
import scipy.optimize


def find_fhv_fvh_coeff_using_fmin(foh, o_array, v_array, h_init=0.0, v_init=0.0, filt_window=51):
    """
    Find the fhv and fvh coefficients by solving nonlinear least squares problem 

    Arguments:

      foh        =  coefficient specifying the fraction of flies transistioning from 
                   "on trap" o[n] to "in trap hidden" h[n] during a given time step 
      o_array    =  array of "on trap" counts giving number of flies on the trap at 

      v_array    =  array of "in trap visible" counts giving number of flies in the trap 
                    which are visible  each time step 

      h_init   =  (optional) initial count of flies in "in trap hidden" state

      v_init   =  (optional) initial count of flies in "in trap visible" state

    Returns:

      fhv      =  coefficient specifying the fraction of flies transistioning from 
                  "in trap hidden" h[n] to "in trap visible" v[n] during a given time 
                  step 

      fvh      =  coefficient specifying the fraction of flies transistioning from 
                  "in trap visible" v[n] to "in trap hidden" h[n] during a given 
                  time step 
    """
    v_array_filt = scipy.signal.medfilt(v_array,filt_window)
    o_array_filt = scipy.signal.medfilt(o_array,filt_window)
    fhv_init, fvh_init = find_fhv_fvh_coeff_using_lstsq(foh, o_array_filt, v_array_filt, 
            h_init=h_init,
            v_init=v_init
            ) 
    param_vec_init  = scipy.array([fhv_init,fvh_init])
    cost_func = create_fhv_fvh_cost_func(foh, o_array, v_array, 
            h_init=h_init, 
            v_init=v_init
            )
    fhv, fvh = scipy.optimize.fmin(cost_func, param_vec_init, disp=0)
    return fhv, fvh 


def find_fhv_fvh_coeff_using_lstsq(foh, o_array, v_array, h_init=0.0, v_init=0.0):
    """
    Find the fhv and fvh coefficients as overdetermined linear system using
    lstsq. Note, This method tends return a biased solution when there is noise
    on the v_array. However, it is very useful for providing an initial guess
    for a more general nonlinear solver.

    Arguments:

      foh        =  coefficient specifying the fraction of flies transistioning from 
                   "on trap" o[n] to "in trap hidden" h[n] during a given time step 
      o_array    =  array of "on trap" counts giving number of flies on the trap at 

      v_array    =  array of "in trap visible" counts giving number of flies in the trap 
                    which are visible  each time step 

      h_init   =  (optional) initial count of flies in "in trap hidden" state

      v_init   =  (optional) initial count of flies in "in trap visible" state

    Returns:

      fhv      =  coefficient specifying the fraction of flies transistioning from 
                  "in trap hidden" h[n] to "in trap visible" v[n] during a given time 
                  step 

      fvh      =  coefficient specifying the fraction of flies transistioning from 
                  "in trap visible" v[n] to "in trap hidden" h[n] during a given 
                  time step 

    """
    n = o_array.shape[0]
    o_array_sum = numpy.zeros((n,))
    o_array_sum[1:] = o_array.cumsum()[:-1]
    h_array =  h_init + v_init + foh*o_array_sum - v_array
    prob_matrix = numpy.zeros((n-1, 2))
    prob_matrix[:,0] = -h_array[:-1]
    prob_matrix[:,1] =  v_array[:-1]
    prob_vector = h_array[1:] - foh*o_array[:-1] - h_array[:-1]
    result = numpy.linalg.lstsq(prob_matrix, prob_vector)
    fhv, fvh = result[0]
    return fhv, fvh 



def find_foh_coeff(count_begin, count_final, o_array):

    """
    Finds the foh coefficient for the linear fly trap model.  The foh coefficient specifies 
    the fraction of flies which transition from "on the trap" to "in the trap, but hidden" 
    during a given time step.


    Arguments:

      count_begin  =  number of flies in trap at the start of the trial 
      count_final  =  number of flies in trap at the end of the trial 
      o_array      =  array of "on trap" counts giving number of flies on the trap at 
                      each time step 
    Returns:

      foh          =  coefficient specifying the fraction of flies transistioning from 
                      "on trap" o[n] to "in trap hidden" h[n] during a given time step 

    """

    foh = (count_final - count_begin)/float(o_array.sum())
    return foh


def find_a_array(foh, o_array):
    """
    Finds the array of fly "on trap arrival" counts for the fly trap model.

    Arguments:

      foh          =  coefficient specifying the fraction of flies transistioning from 
                      "on trap" o[n] to "in trap hidden" h[n] during a given time step 
      o_array      =  array of "on trap" counts giving number of flies on the trap at 
                      each time step 

    Returns:

      a_array      =  array of "on trap arrival" counts given number of flies arriving on 
                      the trap at each time step

    """
    a_array = numpy.zeros(o_array.shape)
    a_array[:-1] = o_array[1:] - (1-foh)*o_array[:-1]
    return a_array



def run_fly_trap_submodel(foh, fhv, fvh, o_array, h_init=0.0, v_init=0.0): 
    """
    Runs  the fly trap submodel simulation ith the given transistion
    coefficients (foh, fhv, fvh) and array of "on trap" fly counts (o_array)
    for each time step.  Note, in this submodel the o_array "on trap" fly
    counts are considered as an inumpyut whereas in the ful model the a_array
    "arrival" fly counts are consider to be an inumpyut. 

    The submodel is as follows: 

    o_array[n+1] = o_array[n] - foh*o_array[n] 
    h_array[n+1] = h_array[n] - fhv*h_array[n] + fvh*v_array[n] + foh*o_array[n]
    v_array[n+1] = v_array[n] + fhv*h_array[n] - fvh*v_array[n] 
    
    where
    
    o_array[n] = # flies on the trap at time step n (inumpyut)
    h_array[n] = # flies in the trap, which are hidden, at time step n
    v_array[n] = # flies in the trap, which are visible, at time step n 

    Arguments:

      foh      =  coefficient specifying the fraction of flies transistioning
                  from "on trap" o[n] to "in trap hidden" h[n] during a given 
                  time step 

      fhv      =  coefficient specifying the fraction of flies transistioning
                  from "in trap hidden" h[n] to "in trap visible" v[n] during 
                  a given time step 

      fvh      =  coefficient specifying the fraction of flies transistioning
                  from "in trap visible" v[n] to "in trap hidden" h[n] during 
                  a given time step 

      o_array  =  array of "on trap" counts giving number of flies on the trap
                  at each time step 

      h_init   =  (optional) initial count of flies in "in trap hidden" state

      v_init   =  (optional) initial count of flies in "in trap visible" state


    Returns:

      h_array  =  array of "in trap hidden" counts giving number of flies in
                  the trap which are not visible at each time step 

      v_array  =  array of "in trap visible" counts giving number of flies in
                  the trap which are visible  each time step 

    """
    num_step = o_array.shape[0]
    h_array = numpy.zeros((num_step,))
    v_array = numpy.zeros((num_step,))
    h_array[0] = h_init
    v_array[0] = v_init
    for i in range(num_step-1):
        o_cur = o_array[i]
        h_cur = h_array[i]
        v_cur = v_array[i]
        h_new = h_cur - fhv*h_cur + fvh*v_cur + foh*o_cur
        v_new = v_cur + fhv*h_cur - fvh*v_cur 
        h_array[i+1] = h_new
        v_array[i+1] = v_new
    return h_array, v_array


def create_fhv_fvh_cost_func(foh, o_array, v_array, h_init=0.0, v_init=0.0):
    """
    Creates the (least squares) cost function for finding the fhv and fhv coefficients.

    Arguments:

      foh      =  coefficient specifying the fraction of flies transistioning from 
                  "on trap" o[n] to "in trap hidden" h[n] during a given time step 

      o_array  =  array of "on trap" counts giving number of flies on the trap at 
                  each time step 

      v_array  =  array of "in trap visible" counts giving number of flies in the trap 
                  which are visible  each time step 

      h_init   =  (optional) initial count of flies in "in trap hidden" state

      v_init   =  (optional) initial count of flies in "in trap visible" state

    Returns:

      cost_func = least squares cost function for finding the fhv and fvh coefficients.  
    
    """
    n = v_array.shape[0]
    o_array_sum = numpy.zeros((n,))
    o_array_sum[1:] = o_array.cumsum()[:-1]
    h_array =  h_init + v_init + foh*o_array_sum - v_array
    def cost_func(f_vec):
        fhv, fvh = f_vec
        h_array_pred, v_array_pred = run_fly_trap_submodel(foh, fhv, fvh, o_array, 
                h_init=h_init, 
                v_init=v_init
                )
        sqr_err = (h_array - h_array_pred)**2 + (v_array - v_array_pred)**2
        return sqr_err.sum()
    return cost_func



# ---------------------------------------------------------------------------------------
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import pykalman
    from simulate import triangle_bump, run_fly_trap_model

    foh = 0.01
    fhv = 0.02
    fvh = 0.01
    num_step = 2000
    noise_level = 0.2

    t_end = 50.0
    t = numpy.linspace(0,t_end,num_step)

    peak_t0 = 3.0
    peak_t1 = 7.0
    peak_max = 3.0
    a_array  = triangle_bump(t,peak_t0,peak_t1,peak_max)

    peak_t0 = 9.0
    peak_t1 = 12.0
    peak_max = 3.0
    a_array  = a_array + triangle_bump(t,peak_t0,peak_t1,peak_max)

    peak_t0 = 12.0
    peak_t1 = 15.0
    peak_max = 2.0
    a_array  = a_array + triangle_bump(t,peak_t0,peak_t1,peak_max)


    o_array, h_array, v_array = run_fly_trap_model(foh, fhv, fvh, a_array) 

    a_array_w_noise = a_array + noise_level*numpy.random.randn(num_step)*a_array.max()
    o_array_w_noise = o_array + noise_level*numpy.random.randn(num_step)*o_array.max()
    h_array_w_noise = h_array + noise_level*numpy.random.randn(num_step)*h_array.max()
    v_array_w_noise = v_array + noise_level*numpy.random.randn(num_step)*v_array.max()

    count_begin = 0.0
    count_final = h_array[-1] + v_array[-1]
    print(count_begin, count_final)

    end_dt = 10.0
    end_mask = t > t_end - end_dt
    count_final = h_array_w_noise[end_mask].mean() + v_array_w_noise[end_mask].mean()

    print(count_begin, count_final)

    foh_est = find_foh_coeff(count_begin, count_final, o_array_w_noise)
    fhv_est, fvh_est = find_fhv_fvh_coeff_using_fmin(foh_est, o_array_w_noise, v_array_w_noise)


    print('foh: {}, foh_est: {}'.format(foh,foh_est))
    print('fhv: {}, fhv_est: {}'.format(fhv,fhv_est))
    print('fvh: {}, fvh_est: {}'.format(fvh,fvh_est))

    trans_mat = numpy.array([
        [1.0-foh_est,          0.0,           0.0],
        [    foh_est,  1.0-fhv_est,       fvh_est],
        [        0.0,      fhv_est,   1.0-fvh_est],
        ])


    obser_mat = numpy.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
        ])

    o_array_var = numpy.diff(o_array_w_noise).var()
    v_array_var = numpy.diff(v_array_w_noise).var()

    print('o var', o_array_var)
    print('v var', v_array_var)

    #trans_cov = 10.0*numpy.diag(numpy.ones((trans_mat.shape[0],)))
    #obser_cov = 1000.0*numpy.diag(numpy.ones((obser_mat.shape[0],)))

    obser_cov = numpy.diag(numpy.array([o_array_var, v_array_var]))

    init_state = numpy.zeros((trans_mat.shape[0],))
    init_state_cov = numpy.diag(numpy.ones((trans_mat.shape[0],)))

    #kalman = pykalman.KalmanFilter( 
    #        transition_matrices = trans_mat, 
    #        observation_matrices = obser_mat, 
    #        transition_covariance = trans_cov, 
    #        observation_covariance = obser_cov, 
    #        initial_state_mean = init_state, 
    #        initial_state_covariance = init_state_cov
    #        )

    kalman = pykalman.KalmanFilter( 
            transition_matrices = trans_mat, 
            observation_matrices = obser_mat, 
            initial_state_mean = init_state, 
            initial_state_covariance = init_state_cov,
            observation_covariance = obser_cov,
            em_vars = ['transition_covariance'],
            )


    n = o_array_w_noise.shape[0]
    data = numpy.zeros((n,2))
    data[:,0] = o_array_w_noise
    data[:,1] = v_array_w_noise

    #state_filt, state_cov =  kalman.smooth(data)
    state_filt, state_cov =  kalman.em(data).smooth(data)

    a_array_est = find_a_array(foh_est, state_filt[:,0])
    acum_array =  a_array.cumsum()
    acum_array_est = a_array_est.cumsum()


    linewidth = 2
    plt.subplot(511)
    #plt.plot(t,a_array_w_noise,'b')
    plt.plot(t,a_array,'r',linewidth=linewidth)
    plt.plot(t,a_array_est,'g',linewidth=linewidth)
    plt.ylabel('arrivals')
    plt.grid('on')

    plt.subplot(512)
    plt.plot(t,o_array_w_noise,'b')
    plt.plot(t,o_array,'r',linewidth=linewidth)
    plt.plot(t,state_filt[:,0],'g',linewidth=linewidth)
    plt.ylabel('on trap')
    plt.grid('on')

    plt.subplot(513)
    plt.plot(t,h_array_w_noise,'b')
    plt.plot(t,h_array,'r',linewidth=linewidth)
    plt.plot(t,state_filt[:,1],'g',linewidth=linewidth)
    plt.ylabel('hidden')
    plt.grid('on')

    plt.subplot(514)
    plt.plot(t,v_array_w_noise,'b')
    plt.plot(t,v_array,'r',linewidth=linewidth)
    plt.plot(t,state_filt[:,2],'g',linewidth=linewidth)
    plt.ylabel('visible')
    plt.grid('on')

    plt.subplot(515)
    plt.plot(t,acum_array,'r',linewidth=linewidth)
    plt.plot(t,acum_array_est,'g',linewidth=linewidth)
    plt.ylabel('cumulative arrivals')
    plt.grid('on')


    plt.xlabel('t')
    plt.show()
