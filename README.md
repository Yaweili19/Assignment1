# Assignment1
Assignment 1 for MACS 30123

## Question 1 (mpi_health.py)
Figure 1 is the computation time.
The speedup is not linear because some portion of the code is serial and run on only rank 0 processor. Besides, increased number of cores increases the time needed to pass message.

![Figure 1](https://github.com/Yaweili19/Assignment1/blob/main/figure1.jpg)

## Question 2 (q2.py)
(A and C) My implementation takes 7.491 seconds to find the optimal rho = 0.0334, where at time = 737.87 health drop below 0, on average.
Figure 2 is below. The x-axis is rho value with y being average time when health drop to 0.
![Figure 2](https://github.com/Yaweili19/Assignment1/blob/main/Fig2.jpg)

## Question 3 (gpu1.py, gpu2.py, gpu3.py)
(A) Original Picture took  0.043 second to run with provided code, and my PyOpencl implementation took 0.047 seconds. 

(B) My PyOpenCL version runs slower than CPU implementation. In my opinion, the to allocate memory and transporting and transforming data back and forth all took time. These added procedures makes the run time in total longer.

(C) It performs progressively better compared to CPU implementation as the amount of data goes up. This is expected because the time to run procedures described above does not scale with the amount of data. As the amount of data increases, the improvement from parrelization becomes more significant.
10 x Picture took  0.315 second to run with provided code, and my PyOpencl implementation took 0.275 seconds. 
20 x Picture took  0.690 second to run with provided code, and my PyOpencl implementation took 0.532 seconds. 
