from __future__ import print_function
import numpy
import pykalman
import param_estimation 


def unscented_kalman_filter_sm(foh, fhv, fvh, o_array, v_array, smooth_param=300.0):

    filter_type = 'additive'
    #filter_type = 'normal'

    trans_mat = get_state_transition_matrix(foh, fhv, fvh)
    obser_mat = get_observation_matrix()

    trans_cov = 1.0*numpy.diag(numpy.ones((trans_mat.shape[0],)))
    # DEVEL
    # -------------------------------------------
    for i in range(1,trans_mat.shape[0]):
        trans_cov[i] = 0.0
    # -------------------------------------------

    obser_cov = smooth_param*numpy.diag(numpy.ones((obser_mat.shape[0],)))

    init_state = numpy.zeros((trans_mat.shape[0],))
    init_state_cov = numpy.diag(numpy.ones((trans_mat.shape[0],)))

    if filter_type == 'additive':

        def state_transition_func(curr_state):
            next_state = numpy.dot(trans_mat, curr_state)
            next_state[next_state < 0] = 0.0
            return next_state

        def observation_func(curr_state):
            obser = numpy.dot(obser_mat, curr_state)
            obser[obser < 0] = 0.0
            return obser

        kalman = pykalman.AdditiveUnscentedKalmanFilter(
                transition_functions = state_transition_func,
                observation_functions = observation_func, 
                transition_covariance = trans_cov, 
                observation_covariance = obser_cov, 
                initial_state_mean = init_state, 
                initial_state_covariance = init_state_cov
                )


    else:

        def state_transition_func(curr_state, noise):
            next_state = numpy.dot(trans_mat, curr_state)
            mask = next_state < 0
            next_state[mask] = 0.0 
            next_state += noise
            return next_state

        def observation_func(curr_state, noise):
            obser = numpy.dot(obser_mat, curr_state)
            mask = obser < 0
            obser[mask] = 0.0
            obser += noise
            return obser

        kalman = pykalman.UnscentedKalmanFilter(
                transition_functions = state_transition_func,
                observation_functions = observation_func, 
                transition_covariance = trans_cov, 
                observation_covariance = obser_cov, 
                initial_state_mean = init_state, 
                initial_state_covariance = init_state_cov
                )


    n = o_array.shape[0]
    data = numpy.zeros((n,2))
    data[:,0] = o_array
    data[:,1] = v_array
    state_filt, state_cov =  kalman.smooth(data)

    return extract_kalman_data(foh, state_filt, state_cov)



def kalman_filter_sm(foh, fhv, fvh, o_array, v_array, smooth_param=300.0):
    
    trans_mat = get_state_transition_matrix(foh, fhv, fvh)
    obser_mat = get_observation_matrix()

    trans_cov = 1.0*numpy.diag(numpy.ones((trans_mat.shape[0],)))
    # DEVEL
    # -------------------------------------------
    for i in range(1,trans_mat.shape[0]):
        trans_cov[i] = 0.0
    # -------------------------------------------
    obser_cov = smooth_param*numpy.diag(numpy.ones((obser_mat.shape[0],)))

    init_state = numpy.zeros((trans_mat.shape[0],))
    init_state_cov = numpy.diag(numpy.ones((trans_mat.shape[0],)))

    kalman = pykalman.KalmanFilter( 
            transition_matrices = trans_mat, 
            observation_matrices = obser_mat, 
            transition_covariance = trans_cov, 
            observation_covariance = obser_cov, 
            initial_state_mean = init_state, 
            initial_state_covariance = init_state_cov
            )

    n = o_array.shape[0]
    data = numpy.zeros((n,2))
    data[:,0] = o_array
    data[:,1] = v_array
    state_filt, state_cov =  kalman.smooth(data)

    return extract_kalman_data(foh, state_filt, state_cov)


def kalman_filter_em(foh, fhv, fvh, o_array, v_array):

    trans_mat = get_state_transition_matrix(foh, fhv, fvh)
    obser_mat = get_observation_matrix()

    o_array_var = numpy.diff(o_array).var()
    v_array_var = numpy.diff(v_array).var()

    obser_cov = numpy.cov(numpy.array([o_array, v_array]))

    init_state = numpy.zeros((trans_mat.shape[0],))
    init_state_cov = numpy.diag(numpy.ones((trans_mat.shape[0],)))

    kalman = pykalman.KalmanFilter( 
            transition_matrices = trans_mat, 
            observation_matrices = obser_mat, 
            initial_state_mean = init_state, 
            initial_state_covariance = init_state_cov,
            observation_covariance = obser_cov,
            em_vars = ['transition_covariance'],
            )

    n = o_array.shape[0]
    data = numpy.zeros((n,2))
    data[:,0] = o_array
    data[:,1] = v_array
    state_filt, state_cov =  kalman.em(data).smooth(data)

    return extract_kalman_data(foh, state_filt, state_cov)


def extract_kalman_data(foh, state_filt, state_cov):

    a_array_filt = param_estimation.find_a_array(foh, state_filt[:,0])
    acum_array_filt = a_array_filt.cumsum()

    o_array_filt = state_filt[:,0]
    h_array_filt = state_filt[:,1]
    v_array_filt = state_filt[:,2]

    o_array_var = state_cov[:,0,0]
    h_array_var = state_cov[:,1,1]
    v_array_var = state_cov[:,2,2]

    a_array_filt = param_estimation.find_a_array(foh, o_array_filt)
    acum_array_filt = a_array_filt.cumsum()

    a_array_var = numpy.zeros(o_array_var.shape)
    a_array_var[:-1] = o_array_var[1:] + (1-foh)*o_array_var[:-1] 
    a_array_var[-1] = o_array_var[-2]

    values  = {
            'o_array'    : o_array_filt,
            'h_array'    : h_array_filt, 
            'v_array'    : v_array_filt,
            'a_array'    : a_array_filt,
            'acum_array' : acum_array_filt,
            }

    variance = {
            'o_array' : o_array_var,
            'h_array' : h_array_var,
            'v_array' : v_array_var,
            'a_array' : a_array_var,
            }

    return values, variance



def get_state_transition_matrix(foh, fhv, fvh):
    mat = numpy.array([
        [1.0-foh,      0.0,       0.0],
        [    foh,  1.0-fhv,       fvh],
        [    0.0,      fhv,   1.0-fvh],
        ])
    return mat

def get_observation_matrix():
    mat = numpy.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
        ])
    return mat


# 2nd form w/ arrivals and cummulative arrivals in kalman filter
# --------------------------------------------------------------------------------------

def kalman_filter_sm2(foh, fhv, fvh, o_array, v_array, smooth_param=300.0):
    
    trans_mat = get_state_transition_matrix2(foh, fhv, fvh)
    obser_mat = get_observation_matrix2()

    trans_cov = 1.0*numpy.diag(numpy.ones((trans_mat.shape[0],)))
    obser_cov = smooth_param*numpy.diag(numpy.ones((obser_mat.shape[0],)))

    init_state = numpy.zeros((trans_mat.shape[0],))
    init_state_cov = numpy.diag(numpy.ones((trans_mat.shape[0],)))

    kalman = pykalman.KalmanFilter( 
            transition_matrices = trans_mat, 
            observation_matrices = obser_mat, 
            transition_covariance = trans_cov, 
            observation_covariance = obser_cov, 
            initial_state_mean = init_state, 
            initial_state_covariance = init_state_cov
            )

    n = o_array.shape[0]
    data = numpy.zeros((n,2))
    data[:,0] = o_array
    data[:,1] = v_array
    state_filt, state_cov =  kalman.smooth(data)

    #return extract_kalman_data2(foh, state_filt, state_cov)


def kalman_filter_em2(foh, fhv, fvh, o_array, v_array):

    trans_mat = get_state_transition_matrix2(foh, fhv, fvh)
    obser_mat = get_observation_matrix2()

    o_array_var = numpy.diff(o_array).var()
    v_array_var = numpy.diff(v_array).var()

    obser_cov = numpy.cov(numpy.array([o_array, v_array]))

    init_state = numpy.zeros((trans_mat.shape[0],))
    init_state_cov = numpy.diag(numpy.ones((trans_mat.shape[0],)))

    kalman = pykalman.KalmanFilter( 
            transition_matrices = trans_mat, 
            observation_matrices = obser_mat, 
            initial_state_mean = init_state, 
            initial_state_covariance = init_state_cov,
            observation_covariance = obser_cov,
            em_vars = ['transition_covariance'],
            )

    n = o_array.shape[0]
    data = numpy.zeros((n,2))
    data[:,0] = o_array
    data[:,1] = v_array
    state_filt, state_cov =  kalman.em(data).smooth(data)

    #return extract_kalman_data2(foh, state_filt, state_cov)


def get_state_transition_matrix2(foh, fhv, fvh):
    mat = numpy.array([
        [1.0,  1.0,     0.0,      0.0,       0.0],
        [0.0,  1.0,     0.0,      0.0,       0.0],
        [0.0,  1.0, 1.0-foh,      0.0,       0.0],
        [0.0,  0.0,     foh,  1.0-fhv,       fvh],
        [0.0,  0.0,     0.0,      fhv,   1.0-fvh],
        ])
    return mat


def get_observation_matrix2():
    mat = numpy.array([
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
    return mat




# Testing
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':

    import sys
    import matplotlib.pyplot as plt
    import pykalman
    import json
    import simulate
    import param_estimation 

    if 0:

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
        a_array  = simulate.triangle_bump(t,peak_t0,peak_t1,peak_max)

        peak_t0 = 9.0
        peak_t1 = 12.0
        peak_max = 3.0
        a_array  = a_array + simulate.triangle_bump(t,peak_t0,peak_t1,peak_max)

        peak_t0 = 12.0
        peak_t1 = 15.0
        peak_max = 2.0
        a_array  = a_array + simulate.triangle_bump(t,peak_t0,peak_t1,peak_max)
        acum_array =  a_array.cumsum()


        o_array, h_array, v_array = simulate.run_fly_trap_model(foh, fhv, fvh, a_array) 

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

        foh_est = param_estimation.find_foh_coeff(count_begin, count_final, o_array_w_noise)
        fhv_est, fvh_est = param_estimation.find_fhv_fvh_coeff_using_fmin(foh_est, o_array_w_noise, v_array_w_noise)

        if 1:
            data, variance = kalman_filter_sm(foh_est, fhv_est, fvh_est, o_array_w_noise, v_array_w_noise, smooth_param=300.0)
        else:
            data, variance = kalman_filter_em(foh_est, fhv_est, fvh_est, o_array_w_noise, v_array_w_noise)

        a_array_est = data['a_array']
        o_array_est = data['o_array']
        h_array_est = data['h_array']
        v_array_est = data['v_array']
        acum_array_est = data['acum_array']

        linewidth = 2
        plt.subplot(511)
        plt.plot(t,a_array,'r',linewidth=linewidth)
        plt.plot(t,a_array_est,'g',linewidth=linewidth)
        plt.ylabel('arrivals')
        plt.grid('on')

        plt.subplot(512)
        plt.plot(t,o_array_w_noise,'b')
        plt.plot(t,o_array,'r',linewidth=linewidth)
        plt.plot(t,o_array_est,'g',linewidth=linewidth)
        plt.ylabel('on trap')
        plt.grid('on')

        plt.subplot(513)
        plt.plot(t,h_array_w_noise,'b')
        plt.plot(t,h_array,'r',linewidth=linewidth)
        plt.plot(t,h_array_est,'g',linewidth=linewidth)
        plt.ylabel('hidden')
        plt.grid('on')

        plt.subplot(514)
        plt.plot(t,v_array_w_noise,'b')
        plt.plot(t,v_array,'r',linewidth=linewidth)
        plt.plot(t,v_array_est,'g',linewidth=linewidth)
        plt.ylabel('visible')
        plt.grid('on')

        plt.subplot(515)
        plt.plot(t,acum_array,'r',linewidth=linewidth)
        plt.plot(t,acum_array_est,'g',linewidth=linewidth)
        plt.ylabel('cumulative arrivals')
        plt.grid('on')

        plt.xlabel('t')
        plt.show()

    else:

        data_filename = '../examples/data/2019_05_08_experiment/all_traps_final_analysis_output.json'
        info_filename = '../examples/data/2019_05_08_experiment/trap_counts.json'

        trap = 'trap_C'
        
        with open(data_filename,'r') as f:
            data = json.load(f)

        trap_data = data[trap]
        t_array = trap_data['seconds since release:']
        o_array = trap_data['flies on trap over time:']
        v_array = trap_data['flies in trap over time:']

        o_array = numpy.array(o_array)
        v_array = numpy.array(v_array)

        with open(info_filename,'r') as f:
            info = json.load(f)
        count_final = info['trap counts'][trap]

        foh_est = param_estimation.find_foh_coeff(0, count_final, o_array)
        fhv_est, fvh_est = param_estimation.find_fhv_fvh_coeff_using_fmin(foh_est, o_array, v_array)

        if 0:
            data, variance = kalman_filter_sm(foh_est, fhv_est, fvh_est, o_array, v_array, smooth_param=100.0)
        else:
            data, variance = kalman_filter_em(foh_est, fhv_est, fvh_est, o_array, v_array)
        
        a_array_est = data['a_array']
        o_array_est = data['o_array']
        h_array_est = data['h_array']
        v_array_est = data['v_array']
        acum_array_est = data['acum_array']

        count_final_filt = h_array_est[-1] + v_array_est[-1]


        print('count_final: ', count_final)
        print('count_final_filt: ', count_final_filt) 

        linewidth = 2
        plt.subplot(511)
        plt.plot(t_array,a_array_est,'g',linewidth=linewidth)
        plt.ylabel('arrivals')
        plt.grid('on')

        plt.subplot(512)
        plt.plot(t_array,o_array,'b',linewidth=linewidth)
        plt.plot(t_array,o_array_est,'g',linewidth=linewidth)
        plt.ylabel('on trap')
        plt.grid('on')

        plt.subplot(513)
        plt.plot(t_array,h_array_est,'g',linewidth=linewidth)
        plt.ylabel('hidden')
        plt.grid('on')

        plt.subplot(514)
        plt.plot(t_array,v_array,'b',linewidth=linewidth)
        plt.plot(t_array,v_array_est,'g',linewidth=linewidth)
        plt.ylabel('visible')
        plt.grid('on')

        plt.subplot(515)
        plt.plot(t_array,acum_array_est,'g',linewidth=linewidth)
        plt.ylabel('cumulative arrivals')
        plt.grid('on')


        plt.xlabel('t')
        plt.show()


