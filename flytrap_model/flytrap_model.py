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
    model, values, variance = fit_and_filter(o_array, v_array, count_final, method, smooth_param)

    return model, values, variance



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
    else:
        values, variance = kalman_filter.kalman_filter_sm(foh_est, fhv_est, fvh_est, o_array, v_array, smooth_param=smooth_param)

    model = {'foh': foh_est, 'fhv': fhv_est, 'fvh': fvh_est}

    return model, values, variance


# Tesing
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    dirname = '../examples/data/2019_05_08_experiment'
    trap = 'C'

    model, values, variance = fit_and_filter_from_dir(dirname,trap)

    print(model)







