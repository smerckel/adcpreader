Backscatter
===========

One of the data that is stored by an ADCP is the echo intensity. This
parameter can be used to determine the validity of the data in terms
of signal-to-noise ratios, see also
:ref:`sec_qualtiy_control_snrlimit`.

The intensity of the acoustic backscatter depends on the size and
abundance of scattering particles in the water column. However,
geometrical spreading and attenuation by the water itself cause the
strength of the signal to be reduced with increasing distance the
acoustic waves travel.

RDI Adcp's store the acoustic backscatter intensity in counts, a
parameter that is linearly proportional to the intensity in dB. For
the Explorer DVL this factor is approximately 0.61 dB/count.

.. math::

   I_b = K (N - N_t)

where :math:`I_b` is the acoustic backscatter intensity expressed in
dB, :math:`K` the conversion factor from counts to dB, :math:`N` the
acoustic backscatter intensity in counts and :math:`N_t` a threshold
count. The acoustic cross section area :math:`\sigma` is given by

.. math::

   \sigma = k_t 10^{I_b/10} r^2 \exp(4 r \alpha)

where :math:`k_t` is a device dependent factor, :math:`r` the distance
to the transducer, and :math:`\alpha` the coefficient of attenuation
due to sea water. The acoustic cross section area can be interpreted
as the backscatter intensity, corrected for the effects of geometrical
spreading and attenuation, and can be related to the abundance and
size of the scattering particles, see also [#f1]_.

The rdi_backscatter module contains a class

.. py:class:: AcousticCrossSection(self, k_t=1e-8, Nt=45, db_per_count=[0.3852]*4)
	      
which takes as arguments ``k_t`` (:math:`k_t`), ``N_t`` (:math:`N_t`) and
``db_per_count`` as a list of factors, one for each beam. This parameter
corresponds to the factor :math:`K`.

If an instance of this class is included in the processing pipeline,
each ping is augmented with a field ``sigma``, containing the
variables ``Sigma1``, ``Sigma2``, ..., ``Sigma<nbeams>``, and
``Sigma_AVG``, where the numeric index denotes the beam number, and
the suffix ``_AVG`` denotes the cell average of all beams.

			     
.. rubric:: References
	    
.. [#f1] L.M. Merckelbach (2006), A model for high-frequency acoustic
	 Doppler current profiler backscatter from suspended sediment
	 in strong currents,
	 Continental Shelf Research 26 1316–1335

