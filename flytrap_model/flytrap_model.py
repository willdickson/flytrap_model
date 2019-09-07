from __future__ import print_function
import json
import kalman_filter
import param_estimation 
import os.path
import numpy


def fit_and_filter_from_dir(dirname, trap, method='em', smooth_param=150.0):

    data_filename = '../examples/data/2019_05_08_experiment/all_traps_final_analysis_output.json'
    info_filename = '../examples/data/2019_05_08_experiment/trap_counts.json'

    data_filename = os.path.join(dirname, 'all_traps_final_analysis_output.json')
    info_filename = os.path.join(dirname, 'trap_counts.json')

    trap_key = 'trap_{}'.format(trap)

    # Load array data
    with open(data_filename,'r') as f:
        data = json.load(f)
    trap_data = data[trap_key]
    t_array = numpy.array(trap_data['seconds since release:'])
    o_array = numpy.array(trap_data['flies on trap over time:'])
    v_array = numpy.array(trap_data['flies in trap over time:'])

    # Load final count data
    with open(info_filename,'r') as f:
        info = json.load(f)
    count_final = info['trap counts'][trap_key]

    # Fit model and filter data
    model, values, variance, inputs = fit_and_filter(o_array, v_array, count_final, method, smooth_param)
    values['t_array'] = t_array
    inputs['t_array'] = t_array

    return model, values, variance, inputs



def fit_and_filter(o_array, v_array, count_final, method='em', smooth_param=150.0):

    """
    Fits flytrap model and applies kalman filter to get filtered values.

    Arguments:

      o_array        =  array of "on trap" counts giving number of flies on the trap at 

      v_array        =  array of "in trap visible" counts giving number of flies in the trap 
                        which are visible  each time step 
      count_final    =  number of flies in trap at the end of the trial 

      method         =  em = expectation maximization, sm = smoothing parameter 

      smooth_param   =  smoothing parameter value. Only used with sm method.


    Returns:

    """

    # Find model parameters using least squares 
    foh_est = param_estimation.find_foh_coeff(0, count_final, o_array)                            # Initial estimate (biased)
    fhv_est, fvh_est = param_estimation.find_fhv_fvh_coeff_using_fmin(foh_est, o_array, v_array)  # Refine estimate

    if method == 'em':
        values, variance = kalman_filter.kalman_filter_em(foh_est, fhv_est, fvh_est, o_array, v_array)
    elif method == 'sm':
        values, variance = kalman_filter.kalman_filter_sm(foh_est, fhv_est, fvh_est, o_array, v_array, smooth_param=smooth_param)
    elif method == 'unscented':
        print('unscented')
        values, variance = kalman_filter.unscented_kalman_filter_sm(foh_est, fhv_est, fvh_est, o_array, v_array, smooth_param=smooth_param)
    else:
        raise ValueError('unknown kalman filtering method')


    # Mega kalman filter (test)
    # -------------------------------------------------------------------------------------------------------
    #kalman_filter.kalman_filter_sm2(foh_est, fhv_est, fvh_est, o_array, v_array, smooth_param=smooth_param)
    #kalman_filter.kalman_filter_em2(foh_est, fhv_est, fvh_est, o_array, v_array)
    # -------------------------------------------------------------------------------------------------------

    values['count_final'] = values['h_array'][-1] + values['v_array'][-1]
    model = {'foh': foh_est, 'fhv': fhv_est, 'fvh': fvh_est}
    inputs = {'o_array': o_array, 'v_array': v_array, 'count_final': count_final}

    return model, values, variance, inputs


# Tesing
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dirname = '../examples/data/2019_05_08_experiment'
    trap = 'C'

    model, values, variance, inputs = fit_and_filter_from_dir(dirname,trap)

    t_array = inputs['t_array']
    o_array = inputs['o_array']
    v_array = inputs['v_array']
    a_array_est = values['a_array']
    o_array_est = values['o_array']
    h_array_est = values['h_array']
    v_array_est = values['v_array']

    o_array_err = 1.0*numpy.sqrt(variance['o_array'])
    h_array_err = 1.0*numpy.sqrt(variance['h_array'])
    v_array_err = 1.0*numpy.sqrt(variance['v_array'])
    a_array_err = 1.0*numpy.sqrt(variance['a_array'])

    acum_array_est = values['acum_array']

    linewidth = 2
    plt.figure(1)
    plt.subplot(311)
    plt.plot(t_array,o_array,'b',linewidth=linewidth)
    plt.plot(t_array,o_array_est,'g',linewidth=linewidth)
    plt.fill_between(t_array,o_array_est+o_array_err,o_array_est-o_array_err,color='g',alpha=0.25)
    plt.ylabel('on trap')
    plt.grid('on')

    plt.subplot(312)
    plt.plot(t_array,h_array_est,'g',linewidth=linewidth)
    plt.fill_between(t_array,h_array_est+h_array_err,h_array_est-h_array_err,color='g',alpha=0.25)
    plt.ylabel('hidden')
    plt.grid('on')

    plt.subplot(313)
    plt.plot(t_array,v_array,'b',linewidth=linewidth)
    plt.plot(t_array,v_array_est,'g',linewidth=linewidth)
    plt.fill_between(t_array,v_array_est+v_array_err,v_array_est-v_array_err,color='g',alpha=0.25)
    plt.ylabel('visible')
    plt.grid('on')
    plt.xlabel('t (sec)')

    plt.figure(2)
    plt.subplot(211)
    plt.plot(t_array,a_array_est,'g',linewidth=linewidth)
    #plt.fill_between(t_array,a_array_est+a_array_err,a_array_est-a_array_err,color='g',alpha=0.25)
    plt.ylabel('arrivals (flies/step)')
    plt.grid('on')
    plt.subplot(212)
    plt.plot(t_array,acum_array_est,'g',linewidth=linewidth)
    plt.ylabel('cumulative arrivals')
    plt.grid('on')
    plt.xlabel('t (sec)')

    plt.show()





