   The gist-geckos Pipeline: An updated version of GIST for use with MUSE data
===============================================================================
This is the gist-GECKOS pipeline, an actively-developed version of the GIST pipeline for use with MUSE data, and 
specially develped for use with the GECKOS and MAUVE surveys. Numerous updates and improvements have been implemented 
on the GIST pipeline. 

Documentation
-------------
For a detailed documentation of the gist-geckos pipeline, including instructions on installation, configuration, and a
tutorial, please see https://geckos-survey.github.io/gist-documentation/

Usage 
-------------

In its default implementation, it extracts stellar kinematics, creates continuum-only and line-only cubes, performs an 
emission-line analysis, derives star formation histories and stellar population properties from full spectral fitting 
as well as via the measurement of absorption line-strength indices. Outputs are easy-to-read 2D maps .fits files of 
various derived parameters, along with best fit spectra for those that want to dive further into the data. 


Citing GIST and the analysis routines
-------------------------------------
If you use this software framework for any publication, please cite Bittner et al. 2019 (A&A, 628, A117;
https://ui.adsabs.harvard.edu/abs/2019A%26A...628A.117B) and include its ASCL entry (http://ascl.net/1907.025) in a
footnote. 

We remind the user to also cite the papers of the underlying analysis techniques and models, if these are used in the
analysis. In the default GIST implementation, these are the adaptive Voronoi tesselation routine (Cappellari & Copin
2003), the penalised pixel-fitting method (pPXF; Cappellari & Emsellem 2004; Cappellari 2017), the pyGandALF routine
(Sarzi et al. 2006; Falcon-Barroso et al. 2006; Bittner et al. 2019), the line-strength measurement routines (Kuntschner
et al. 2006; Martin-Navarro et al. 2018), and the MILES models included in the tutorial (Vazdekis et al. 2010). 


Disclaimer
----------
Although we provide this software as a convenient, all-in-one framework for the analysis of integral-field spectroscopic
data, it is of fundamental importance that the user understands exactly how the involved analysis methods work. We warn
that the improper use of any of these analysis methods, whether executed within the framework of the GIST or not, will
likely result in spurious or erroneous results and their proper use is solely the responsibility of the user. Likewise,
the user should be fully aware of the properties of the input data before intending to derive high-level data products.
Therefore, this software framework should not be simply adopted as a black-box. To this extend, we urge any user to get
familiar with both the input data and analysis methods, as well as their implementation.





