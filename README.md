# The Algorithmic Pulsar Timer, APT
APT is an algorithm designed to time isolated pulsars given an input .par and .tim file. The algorithm uses covariances between fit parameters to project possible models forward in time. By combining these predictive models with the statistical F-Test, along with several features such as phase wrapping, checking for bad data, and choosing appropriate starting TOAs, the algorithm encodes the decisions a scientist makes when fitting a pulsar into an easy-to-run script with an average runtime of several minutes. APT was tested on 100 simulated systems, created using the included simdata.py script, and solved 97% of the systems. APT fits for the four main paramaters, F0 (spin), RAJ (Right Ascension), DECJ (Declination), and F1 (spindown). 

APT is described in the paper *Algorithmic Pulsar Timing*, available on the ArXiv at ***. 
APT is dependent on the pulsar timing package PINT, available at (https://github.com/nanograv/PINT) and described in the paper *PINT: A Modern Software Package for Pulsar Timing*, available at https://arxiv.org/abs/2012.00074
