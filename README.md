# Adaptive Systems

Project Description is available at [my university page](http://www.cs.toronto.edu/~shaktik/) under Proactive Troubleshooting [Aug2018-Present].

I will update this readme as I get more time, the individual files however have an extensive documentation.

```experiment_sin_stability.py```
 
 This file discusses an implementation of stabilizing a noisy sin function. The noise here is modelled as a gaussian with a choice of mean and variance.
A reward structure promoting the stability is defined. The Utility values are learnt using Generalized Policy Iteration method. There are _just 2 steps_ in the entire file,
1. Policy Evaluation U &rarr; U<sup>π</sup>

    Return(s) = &sum;&gamma;<sup>t</sup>R(S<sub>t</sub>)

    Q<sup>&pi;</sup>(s, a) = E {Return<sub>t</sub> | s<sub>t</sub>, a<sub>t</sub>}

    U<sup>&pi;</sup>(s) = E [ &sum;&gamma;<sup>t</sup>R(S<sub>t</sub>) ]

2. Policy Improvement π &rarr; greedy(U)

    &pi;(s) = argmax<sub>a</sub> Q(s,a)

If you would like to know more about this research feel free to drop me an email.
