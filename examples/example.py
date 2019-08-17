
import numpy
import flytrap_model
import matplotlib.pyplot as plt


#dirname = './data/2019_05_08_experiment'
#trap = 'C'

dirname = './data/2017_10_26_experiment'
trap = 'G'

#model, values, variance, inputs = flytrap_model.fit_and_filter_from_dir(dirname,trap,method='sm',smooth_param=0.01)
model, values, variance, inputs = flytrap_model.fit_and_filter_from_dir(dirname,trap)

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
plt.ylabel('arrivals (flies/step)')
plt.grid('on')
plt.subplot(212)
plt.plot(t_array,acum_array_est,'g',linewidth=linewidth)
plt.ylabel('cumulative arrivals')
plt.grid('on')
plt.xlabel('t (sec)')

plt.show()





