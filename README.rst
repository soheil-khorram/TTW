.. -*- mode: rst -*-

Trainable Time Warping (TTW)
============================

Matlab implementation of TTW, a time-warping algorithm that estimates the DTW averaging solution in linear time and space complexity using a new deformable convolutional kernel, shifted sinc kernel.

.. image:: ttw_results.png

What is this repository?
------------------------

This repository contains the MATLAB code of three dynamic time warping (DTW) averaging algorithms: generalized time warping (GTW) [1], trainable time warping (TTW) [2] and fast learnable time warping (FLTW) [2]. All these algorithms have linear time and space complexity with respect to the length of input time-series. 

GTW
-----

Generalized time warping (GTW) is a DTW averaging algorithm that can align multiple time-series with linear complexity in the length of time-series [1]. GTW approximates the optimal temporal warping by linearly combining a fixed set of monotonic basis functions. Authors of [1] introduced a Gauss-Newton-based procedure to learn the weights of the basis functions. However, in the cases where the temporal relationship between the time-series is complex, GTW requires a large number of complex basis functions to be effective; defining these basis functions is very difficult [2].

avg_gtw.m provides a function for GTW averaging. The function takes two inputs, X and max_iter_num, and generates three outputs,
y, XS and tau. 

* Inputs
      - X: input time-series; it is an N-by-T matrix where N is the number of time-series and T is the length of them. Note that this function assumes all time-series have the same length. If your time-series have different lengths, use an interpolation technique (e.g., spline function) to make them equi-length.
      - max_iter_num: maximum number of iterations in our gradient-based optimization algorithm.

* Outputs
      - y: final average signal. It is a 1-by-T vector.
      - XS: synchronized time-series. y is equal to mean(XS, 1).
      - tau: learned warping functions. It is an N-by-T matrix. tau(n, :) is the warping function used to obtain XS(n, :) from X(n, :).


TTW
-----

TTW offers another solution to the problem of aligning multiple time-series. It leverages a new convolutional kernel (shifted sinc kernel) that can apply non-linear temporal warping functions to time-series. It uses the kernel along with a gradient-based optimization technique to perform the alignment. TTW is linear with the length of the input time-series. The experiments in [2] show that TTW provides an effective time alignment; it outperforms the GTW on DTW averaging and nearest centroid classification tasks. 

avg-ttw is the function that we designed to perform the averaging using the TTW algorithm. It takes Three inputs (X, order, max_iter_num) and returns three outputs (y, XS, tau):

* Inputs
      - X: input time-series; it is an N-by-T matrix where N is the number of time-series and T is the length of them. Note that this algorithm assumes all time-series have the same length. If your time-series have different lengths, use an interpolation technique (e.g., spline function) to make them equi-length.
      - order: order of the DST used to model the warping functions. With higher order, we consider more details of the input signals and we learn more complex warping functions. However, a high order value will cause a more complex optimization landscape, which increases the likelihood of converging to a weak local optimum point.
      - max_iter_num: maximum number of iterations in our gradient-based optimization algorithm.

* Outputs
      - y: final average signal. It is a 1-by-T vector.
      - XS: synchronized time-series. y is equal to mean(XS, 1).
      - tau: learned warping functions. It is an N-by-T matrix. tau(n, :) is the warping function used to obtain XS(n, :) from X(n, :).

FLTW
-----

Fast learnable time warping (FLTW) is an improved version of the TTW algorithm. It is faster than TTW and it provides a better temporal warping. It uses shifted triangular filter instead of the shifted sinc kernel. Our experiments show that the FLTW algorithm significantly outperforms both GTW and TTW algorithms in time-series averaging and time-series classification tasks. avg_fltw.m provides a function for this technique. Its usage is quite similar to the avg_ttw.

References
----------

.. [1] Feng Zhou and Fernando De la Torre,
       *“Generalized time warping for multi-modal alignment of human motion”*,
       CVPR, 2012. [`PDF <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.227.6175&rep=rep1&type=pdf>`_]

.. [2] Soheil Khorram, Melvin McInnis and Emily Mower Provost,
       *“Trainable Time Warping: Aligning Time-Series in the Continuous-Time Domain”*,
       ICASSP, 2019. [`PDF <https://arxiv.org/pdf/1903.09245.pdf>`_]

Author
------

- Soheil Khorram, 2019
