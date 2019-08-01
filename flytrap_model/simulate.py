import numpy

def run_fly_trap_model(foh, fhv, fvh, a_array, o_init=0.0, h_init=0.0, v_init=0.0): 
    """
    Runs  the fly trap model simulation with the given transistion coefficients 
    (foh, fhv, fvh) and array of "fly arrival" counts (a_array)

    The model is as follows: 

    o_array[n+1] = o_array[n] - foh*o_array[n] + a_array[n] 
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
                  from "on trap visible" o[n] to "in trap hidden" h[n] during 
                  a given time step 

      a_array  =  array of "arrivals" giving number of flies arriving on the 
                  trap at each time step 

      o_init   =  (optional) initial count of flies in "on trap" state

      h_init   =  (optional) initial count of flies in "on trap hidden" state

      v_init   =  (optional) initial count of flies in "on trap visible" state


    Returns:

      o_array  =  array of "on trap" counts giving number of flies on the trap
                  at each time step 

      h_array  =  array of "in trap hidden" counts giving number of flies in
                  the trap which are not visible at each time step 

      v_array  =  array of "in trap visible" counts giving number of flies in
                  the trap which are visible  each time step 

    """
    num_step = a_array.shape[0]
    o_array = numpy.zeros((num_step,))
    h_array = numpy.zeros((num_step,))
    v_array = numpy.zeros((num_step,))
    o_array[0] = o_init
    h_array[0] = h_init
    v_array[0] = v_init
    for i in range(num_step-1):
        o_cur = o_array[i]
        h_cur = h_array[i]
        v_cur = v_array[i]
        o_new = o_cur - foh*o_cur + a_array[i]
        h_new = h_cur - fhv*h_cur + fvh*v_cur + foh*o_cur
        v_new = v_cur + fhv*h_cur - fvh*v_cur 
        o_array[i+1] = o_new
        h_array[i+1] = h_new
        v_array[i+1] = v_new
    return o_array, h_array, v_array


def triangle_bump(t_array, t_start, t_stop, peak_value):
    """
    Creates a symmetrical triangular bump starting at t_start and ending t_stop with 
    peak height of peak_values.

    Arguments:
      
      t_array     =  array of time points
      t_start     =  time at which bump starts
      t_stop      =  time at which bump stops 
      peak_value  =  the peak height of the bump. 

    Returns:

      value_array = array of function values.

    """

    dt = t_array[1] - t_array[0]
    t_peak = 0.5*(t_start + t_stop)
    rate = peak_value/(t_peak - t_start)
    value_list = []
    value = 0.0
    for t in t_array:
        value_list.append(value)
        if t > t_start and t <= t_peak:
            value += rate*dt
        elif t > t_peak and t <= t_stop:
            value -= rate*dt
        else:
            value = 0.0
    value_array = numpy.array(value_list)
    return value_array


# ---------------------------------------------------------------------------------------
if __name__  == '__main__':

    import matplotlib.pyplot as plt

    foh = 0.01
    fhv = 0.02
    fvh = 0.01

    t = numpy.linspace(0,30,500)
    t0 = 2.0
    t1 = 5.0
    peak = 4.0

    a_array  = triangle_bump(t,t0,t1,peak)
    o_array, h_array, v_array = run_fly_trap_model(foh, fhv, fvh, a_array) 
        
    plt.subplot(411)
    plt.plot(t,a_array)
    plt.ylabel('arrivals')
    plt.grid('on')

    plt.subplot(412)
    plt.plot(t,o_array)
    plt.ylabel('on trap')
    plt.grid('on')

    plt.subplot(413)
    plt.plot(t,h_array)
    plt.ylabel('hidden')
    plt.grid('on')

    plt.subplot(414)
    plt.plot(t,v_array)
    plt.ylabel('visible')
    plt.grid('on')

    plt.xlabel('t')
    plt.show()
