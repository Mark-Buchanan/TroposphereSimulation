# Program simulates the atmosphere to get the Boltzmann and Maxwell-Boltzmann
# distributions for a dry atmosphere composed of Nitrogen and Oxygen.  Uses
# various different random number generators for the simulation to gauge
# their usefulness.

# imports and setup
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# defining physical constants
k_B = sci.Boltzmann               # Boltzmann constant J / K
g = 9.81                          # gravitational constant m / s^2
m_N2 = 2 * 14.0067 * 1.6605e-27   # mass N2 molecule Kg
m_O2 = 2 * 15.99 * 1.6605e-27     # mass O2 molecule Kg
T_bot = 290                       # temp at sea level K
T_top = 222                       # temp at top of troposphere K
num_T = 20                        # number of temperature regions


# #############################################################################
# ############### CAN CHANGE SIMULATION PARAMETERS HERE #######################
# #############################################################################
# defining simulation constants
# NOTE: If you're testing it, you should only change N, iterations, and the
#       chosen random number generator below (line ~185).  Simulations take a
#       long time for the default N, iterations given (reduce both by 1-2 
#       orders of magnitude)
N = 5000                          # number molecules to simulate
iterations = 25000                # number of iterations of monte carlo sim
prop_N2 = 0.79                    # fractional percentage of atmosphere of N2
prop_O2 = 0.21                    # fractional percentage of atmosphere of O2
step_h = 250                      # abs val of max step size for h simulation
step_v = 50                       # abs val of max step size for v simulation

# seeds for the various random number generators
np.random.seed(17)
ran_num = np.asarray([17.0])


###############################################################################
# function definitions

# random number generator using numpy implementation
def rng_np():
    return np.random.random()


# random number generator using the linear congruential method modified from 
# the textbook implementation
def rng_lc():
    a = 1664525
    c = 1013904223
    m = 4294967296
    ran_num[0] = (a * ran_num[0] + c) % m
    return ran_num[0] / m


# random number generator using a bad implementation of the linear
# congruential method modified from the textbook implementation
def rng_lc_bad():
    a = 167
    c = 1013
    m = 4297
    ran_num[0] = (a * ran_num[0] + c) % m
    return ran_num[0] / m


# random number generator using the inverse congruential method
def rng_ic():
    a = 57
    c = 96
    m = 139
    # ensure not dividing by zero
    if ran_num[0] != 0:
        ran_num[0] = (c + a * ran_num[0]**(m-2)) % m
    else:
        ran_num[0] = c
        
    return ran_num[0] / m


# function that initializes the sample atmosphere to N molecules
def atmosphere_init(N, rng):
    # initialize arrays, temperature take to decrease linearly from 0 to 19 km
    # from 17 degrees C to -51 degrees C, v_arr is x, y, z components and total
    h_arr = np.zeros(N)
    T_arr = np.linspace(T_bot, T_top, num_T)
    E_arr = np.zeros(iterations)
    v_arr = np.zeros([4, N])
    K_arr = np.zeros(iterations)
    
    # loop populating height and speed arrays between 0 and step_h metres and
    # step_v m/s respectively using the rng function specified
    for i in range(N):
        h_arr[i] = step_h * rng()
        v_arr[0, i] = step_v * rng()
        v_arr[1, i] = step_v * rng()
        v_arr[2, i] = step_v * rng()
        v_arr[3, i] = np.sqrt(v_arr[0, i]**2 + v_arr[1, i]**2 + v_arr[2, i]**2)
        
    return h_arr, T_arr, E_arr, v_arr, K_arr

# helper function for the metropolis algorithm that takes in various physical
# parameters as well as the array start/stop indices, and the random number
# generator 
# returns updated height and speed arrays after 1 round of simulation
def metro_loop(h_arr, T_arr, v_arr, start, end, m, rng):
    # main loop to perform height and speed calculations for the 
    # end-start molecules
    for i in range(start, end):
        # calculate initial 'energies' (not actually energy, to reduce
        # numerical errors later) and candidate heights, speeds
        E_i = m * g * h_arr[i] / k_B
        h_new = h_arr[i] + step_h * (2 * rng() - 1)

        v_i_sq = v_arr[0, i] ** 2 + v_arr[1, i] ** 2 + v_arr[2, i] ** 2
        v_new_x = v_arr[0, i] + step_v * (2 * rng() - 1)
        v_new_y = v_arr[1, i] + step_v * (2 * rng() - 1)
        v_new_z = v_arr[2, i] + step_v * (2 * rng() - 1)
        K_i = (m * v_i_sq) / (2 * k_B)

        # ensure molecule doesn't try to leave troposphere
        while h_new > 19000 or h_new < 0:
            h_new = h_arr[i] + step_h * (2 * rng() - 1)

        # candidate step's final energies and variables for conciseness
        T_f = T_arr[int(h_new / 1000)]
        E_f = m * g * h_new / k_B
        del_E = E_f - E_i

        v_new_sq = v_new_x ** 2 + v_new_y ** 2 + v_new_z ** 2
        K_f = (m * v_new_sq) / (2 * k_B)
        del_K = K_f - K_i

        # accept new lower energy heights or higher energy states if they
        # satisfy probability of acceptance
        if del_E <= 0 or (del_E > 0 and rng() <= np.exp(-del_E / T_f)):
            h_arr[i] = h_new

        # accept lower energy speeds or higher energy states if they
        # satisfy probability of acceptance
        if del_K <= 0 or (del_K > 0 and rng() <= np.exp(-del_K / T_f)):
            v_arr[3, i] = np.sqrt(v_new_sq)

    return h_arr, v_arr


# master function that calls helper functions and performs the full monte carlo
# simulation using the metropolis algorithm
def metropolis(N, iterations, rng):
    # initialize height, temperature, potential energy, speed, and kinetic
    # energy arrays as well as the number of Nitrogen/Oxygen molecules
    h_arr, T_arr, E_arr, v_arr, K_arr = atmosphere_init(N, rng)
    N_N2 = int(N * prop_N2)
    N_O2 = int(N * prop_O2)
    
    # main loop performing iterations num iterations of metropolis algorithm
    for i in range(iterations):
        # simulate each of the N particles for this iteration of the metropolis
        # algorithm
        h_arr, v_arr = metro_loop(h_arr, T_arr, v_arr, 0, N_N2, m_N2, rng)
        h_arr, v_arr = metro_loop(h_arr, T_arr, v_arr, N - N_O2, N, m_O2, rng)
            
        # calculate the total energy of all the particles each iteration
        E_arr[i] = g * (m_N2 * sum(h_arr[0:N_N2]) + m_O2 * sum(h_arr[N_N2:]))
        K_arr[i] = (m_N2 * sum((v_arr[3, :N_N2])**2) +
                    m_O2 * sum((v_arr[3, N_N2:])**2)) / 2
        
        # progress report for every 10% of simulation completion
        if i % int(iterations / 10) == 0:
            print(str(i) + ' simulations done out of ' + str(iterations))
    
    # return final values of height, speed and all values of the energies    
    return h_arr, E_arr, v_arr[3, :], K_arr


###############################################################################
# NOTE: This function call runs the simulation with the desired parameters.
#       N and iterations are changed in the constants section, you can change
#       the random number generator below to one of the option functions all
#       starting with rng.  Also decrease N and iterations to run in a
#       reasonable amount of time.
ran_num_gen = rng_np
h_arr, E_arr, v_arr, K_arr = metropolis(N, iterations, ran_num_gen)


###############################################################################
# plotting code

# define bins for histograms
h_bins = np.linspace(0, max(h_arr), 20)
v_bins = np.linspace(0, max(v_arr), 20)

# plot histogram of heights and the number of their occurrences
plt.figure(1)
plt.hist(h_arr[0: int(N * prop_N2)], bins=h_bins, label='$N_2$')
plt.hist(h_arr[int(N * prop_N2):], bins=h_bins, label='$O_2$')
plt.title('Number of Occurrences of Height for ' + str(N) +
          ' $N_2$, $O_2$ Molecules \nAfter ' + str(iterations) +
          ' Iterations with ' + str(len(h_bins)) + ' Bins')
plt.xlabel('Height ($m$)')
plt.ylabel('Number of events')
plt.legend()
plt.savefig('project_1a.pdf')
plt.show()

# plot histogram of a PDF for the heights
plt.figure(2)
plt.hist(h_arr[0: int(N * prop_N2)], bins=h_bins, density=True, label='$N_2$')
plt.hist(h_arr[int(N * prop_N2):], bins=h_bins, density=True, label='$O_2$', 
         lw=3, fc=(1, 0, 0, 0.5))
plt.title('Height PDF for ' + str(N) + ' $N_2$, $O_2$ Molecules After ' +
          str(iterations) + '\nIterations with ' + str(len(h_bins)) + ' Bins')
plt.xlabel('Height ($m$)')
plt.ylabel('Probability density function')
plt.legend()
plt.savefig('project_1b.pdf')
plt.show()

# plot of the energy after each iteration of the simulation
plt.figure(3)
plt.plot(E_arr)
plt.title('Total Potential Energy vs. Number of Iterations for ' + str(N) +
          '\n$N_2$, $O_2$ Molecules After ' + str(iterations) + ' Iterations')
plt.xlabel('Number of iterations')
plt.ylabel('Total energy of all molecules')
plt.savefig('project_1c.pdf')
plt.show()

# plot histogram of heights and the number of their occurrences
plt.figure(4)
plt.hist(v_arr[0: int(N * prop_N2)], bins=v_bins, label='$N_2$')
plt.hist(v_arr[int(N * prop_N2):], bins=v_bins, label='$O_2$')
plt.title('Number of Occurrences of Speed for ' + str(N) +
          ' $N_2$, $O_2$ Molecules \nAfter ' + str(iterations) +
          ' Iterations with ' + str(len(v_bins)) + ' Bins')
plt.xlabel('Speed ($m/s$)')
plt.ylabel('Number of events')
plt.legend()
plt.savefig('project_1d.pdf')
plt.show()

# plot histogram of a PDF for the heights
plt.figure(5)
plt.hist(v_arr[0: int(N * prop_N2)], bins=v_bins, density=True, label='$N_2$')
plt.hist(v_arr[int(N * prop_N2):], bins=v_bins, density=True, label='$O_2$', 
         lw=3, fc=(1, 0, 0, 0.5))
plt.title('Speed PDF for ' + str(N) + ' $N_2$, $O_2$ Molecules After ' +
          str(iterations) + '\nIterations with ' + str(len(v_bins)) + ' Bins')
plt.xlabel('Speed ($m/s$)')
plt.ylabel('Probability density function')
plt.legend()
plt.savefig('project_1e.pdf')
plt.show()

# plot of the energy after each iteration of the simulation
plt.figure(6)
plt.plot(K_arr, '.', markersize=3)
plt.title('Total Kinetic Energy vs. Number of Iterations for ' + str(N) +
          '\n$N_2$, $O_2$ Molecules After ' + str(iterations) + ' Iterations')
plt.xlabel('Number of iterations')
plt.ylabel('Total kinetic energy of all molecules')
plt.savefig('project_1f.pdf')
plt.show()
