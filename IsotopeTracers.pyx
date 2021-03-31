"""
This is the Stable water isotope tracer components of ipycles, will activate when namelist['IsotopeTracer']=true
"""
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
cimport ReferenceState

from libc.math cimport log, exp
from NetCDFIO cimport NetCDFIO_Stats
import cython

cimport numpy as np
import numpy as np

def IsotopeTracersFactory(namelist):
    try:
        use_isotope_tracers = namelist['isotopetracers']['use_tracers']
    except:
        use_isotope_tracers = False
    if use_isotope_tracers:
        try:
            tracer_scheme = namelist['isotopetracers']['scheme']
            if tracer_scheme == 'No microphysics':
                return IsotopeTracers_NoMicrophysics(namelist)
            else:
                print('IsotopeTracers scheme is not recognized, using IsotopeTracersNone')
                return IsotopeTracersNone()
        except:
            return IsotopeTracersNone()
    else:
        return IsotopeTracersNone()

cdef class IsotopeTracersNone:
    def __init__(self):
        return
    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with isotopenone')
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

cdef class IsotopeTracers_NoMicrophysics:
    def __init__(self, namelist):
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with isotopehave')
        PV.add_variable('r_vapor', 'permill','R_vapor','Isotope ratio of water vapor (HDO)','sym', "scalar", Pa)
        DV.add_variables('r_liquid', 'permill','R_cloud_liquid_water','Isotope ratio of cloud liquid water (HDO)','sym', Pa)
        
        NS.add_profile('r_vapor', Gr, Pa, units=r'permill', nice_name='R_vapor', desc='Isotope ratio of water vapor (HDO)')
        NS.add_profile('r_liquid', Gr, Pa, units=r'permill', nice_name='R_vapor', desc='Isotope ratio of water vapor (HDO)')
        
        return
        
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t i, j, k, ijk
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t ishift
            Py_ssize_t jshift 

            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t r_vapor_varshift = PV.get_varshift(Gr,'r_vapor')
            Py_ssize_t r_liquid_varshift = DV.get_varshift(Gr,'r_liquid')
            Py_ssize_t qt_varshift = PV.get_varshift(Gr, 'qt')
            double alpha_eq

        # Fractionation when the cell has liquid water
        with cython.boundscheck(False):
            with nogil:
                for i in xrange(Gr.dims.nlg[0]):
                    ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
                    for j in xrange(Gr.dims.nlg[1]):
                        jshift = j * Gr.dims.nlg[2]
                        for k in xrange(Gr.dims.nlg[2]):
                            ijk = ishift + jshift + k
                            alpha_eq = equilibrium_fractionation_factor(DV.values[t_shift + ijk])
                        # condensation as ql>0, assuming all vapor condense
                            if DV.values[ql_shift + ijk] > 0:
                                DV.values[r_liquid_varshift + ijk] += PV.values[r_vapor_varshift + ijk] * alpha_eq
                                PV.values[r_vapor_varshift + ijk] = 0
                        # evaporation as ql = 0, assuming all liquid evaporate
                            else:
                                if DV.values[r_liquid_varshift + ijk] > 0:
                                    PV.values[r_vapor_varshift + ijk] += DV.values[r_liquid_varshift + ijk] / alpha_eq
                                    DV.values[r_liquid_varshift + ijk] = 0

    
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t r_vapor_varshift = PV.get_varshift(Gr,'r_vapor')
            Py_ssize_t r_liquid_varshift = DV.get_varshift(Gr,'r_liquid')

        tmp = Pa.HorizontalMean(Gr, &PV.values[r_vapor_varshift])
        NS.write_profile('r_vapor', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &DV.values[r_liquid_varshift])
        NS.write_profile('r_liquid', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

# calculate equilibrium fractionation factor using given temperature, based on emperical equation from Majoube 1971
cdef double equilibrium_fractionation_factor(double t) nogil:
    cdef:
        double alpha_eq
    
    alpha_eq = exp( 1137/(t*t) - 0.4156/t - 2.0667e-3) 
    return alpha_eq
