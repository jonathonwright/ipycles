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
        # Prognostic variable: qt_iso, total water isotopic specific humidity, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_iso', 'kg/kg','qt_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        # Diagnostic variables: ql_iso, cloud liquid water isotopic specific humidity, defined as the ratio of isotopic mass of H2O18 in cloud droplets to moist air
        #                       qv_iso, vapor water isotopic specific humidity, defined as the ratio of isotopic mass of H2O18 in water vapor to moist air  
        #                       r_ql_iso, isotope ratio of H2O18 in liqudi water, defined as ql_iso/ql
        #                       r_qv_iso, isotope ratio of H2O18 in water vapor, defined as qv_iso/qv
        PV.add_variable('ql_iso', 'kg/kg','ql_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('qv_iso', 'kg/kg','qv_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        DV.add_variables('r_ql_iso', 'permil','isotope_ratio_liquid','isotope ratio of liqudi water','sym', Pa)
        DV.add_variables('r_qv_iso_in_cloud', 'permil','isotope_ratio_vapor_in_cloud','isotope ratio of water vapor during fractionation','sym', Pa)
        DV.add_variables('r_qv_iso', 'permil','isotope_ratio_vapor','isotope ratio of water vapor','sym', Pa)
        
        NS.add_profile('qt_iso', Gr, Pa, units=r'kg/kg', nice_name='qt_isotope', desc='Total water isotopic specific humidity')
        NS.add_profile('ql_iso', Gr, Pa, units=r'kg/kg', nice_name='ql_isotope', desc='Cloud liquid water isotopic specific humidity')
        NS.add_profile('qv_iso', Gr, Pa, units=r'kg/kg', nice_name='qv_isotope', desc='Vapor water isotopic specific humidity')
        NS.add_profile('r_ql_iso', Gr, Pa, units=r'permil', nice_name='isotope_ratio_liquid', desc='isotope ratio of liqudi water')
        NS.add_profile('r_qv_iso_in_cloud', Gr, Pa, units=r'permil', nice_name='isotope_ratio_vapor_in_cloud', desc='isotope ratio of water vapor during fractionation')
        NS.add_profile('r_qv_iso', Gr, Pa, units=r'permil', nice_name='isotope_ratio_vapor', desc='isotope ratio of water vapor')
        
        return
        
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t i, j, k, ijk
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t ishift
            Py_ssize_t jshift 
            Py_ssize_t qv_varshift = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_varshift = DV.get_varshift(Gr,'ql')
            Py_ssize_t t_varshift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_iso_varshift = PV.get_varshift(Gr,'qt_iso')
            Py_ssize_t ql_iso_varshift = PV.get_varshift(Gr,'ql_iso')
            Py_ssize_t qv_iso_varshift = PV.get_varshift(Gr,'qv_iso')
            Py_ssize_t r_ql_iso_varshift = DV.get_varshift(Gr,'r_ql_iso')
            Py_ssize_t r_qv_iso_varshift = DV.get_varshift(Gr,'r_qv_iso')
            Py_ssize_t r_qv_iso_in_cloud_varshift = DV.get_varshift(Gr,'r_qv_iso_in_cloud')
            Py_ssize_t qt_varshift = PV.get_varshift(Gr, 'qt')
            double alpha_eq
            double q_li
            double q_vi

        # Fractionation during phase changes
        with cython.boundscheck(False):
            with nogil:
                for i in xrange(Gr.dims.nlg[0]):
                    ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
                    for j in xrange(Gr.dims.nlg[1]):
                        jshift = j * Gr.dims.nlg[2]
                        for k in xrange(Gr.dims.nlg[2]):
                            ijk = ishift + jshift + k
                            alpha_eq = 1 / equilibrium_fractionation_factor(DV.values[t_varshift + ijk])
                            if DV.values[ql_varshift + ijk] > 0:
                                q_li = q_li_equilibrium_fractionation(PV.values[qt_iso_varshift + ijk], DV.values[qv_varshift + ijk], DV.values[ql_varshift + ijk], alpha_eq)
                                PV.values[ql_iso_varshift + ijk] = q_li
                                DV.values[r_ql_iso_varshift + ijk] = q_li / DV.values[ql_varshift + ijk]
                                q_vi = PV.values[qt_iso_varshift + ijk] - q_li
                                PV.values[qv_iso_varshift + ijk] = q_vi
                                DV.values[r_qv_iso_varshift + ijk] = q_vi / DV.values[qv_varshift + ijk]
                                DV.values[r_qv_iso_in_cloud_varshift + ijk] = q_vi / DV.values[qv_varshift + ijk]
                                
                            else:
                                q_li = 0.0
                                DV.values[r_ql_iso_varshift + ijk] = 0.0
                                PV.values[ql_iso_varshift + ijk] = q_li
                                q_vi = PV.values[qt_iso_varshift + ijk] - q_li
                                PV.values[qv_iso_varshift + ijk] = q_vi
                                DV.values[r_qv_iso_varshift + ijk] = q_vi / DV.values[qv_varshift + ijk]
                                DV.values[r_qv_iso_in_cloud_varshift + ijk] = 0.0
    
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t qt_iso_varshift = PV.get_varshift(Gr,'qt_iso')
            Py_ssize_t ql_iso_varshift = PV.get_varshift(Gr,'ql_iso')
            Py_ssize_t qv_iso_varshift = PV.get_varshift(Gr,'qv_iso')
            Py_ssize_t r_ql_iso_varshift = DV.get_varshift(Gr,'r_ql_iso')
            Py_ssize_t r_qv_iso_varshift = DV.get_varshift(Gr,'r_qv_iso')
            Py_ssize_t r_qv_iso_in_cloud_varshift = DV.get_varshift(Gr,'r_qv_iso_in_cloud')

        tmp = Pa.HorizontalMean(Gr, &PV.values[qt_iso_varshift])
        NS.write_profile('qt_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &PV.values[ql_iso_varshift])
        NS.write_profile('ql_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &PV.values[qv_iso_varshift])
        NS.write_profile('qv_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   

        tmp = Pa.HorizontalMean(Gr, &DV.values[r_ql_iso_varshift])
        NS.write_profile('r_ql_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &DV.values[r_qv_iso_varshift])
        NS.write_profile('r_qv_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &DV.values[r_qv_iso_in_cloud_varshift])
        NS.write_profile('r_qv_iso_in_cloud', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

# calculate equilibrium fractionation factor using given temperature, based on emperical equation from Majoube 1971
cdef double equilibrium_fractionation_factor(double t) nogil:
    cdef double alpha_eq
    alpha_eq = exp( 1137/(t*t) - 0.4156/t - 2.0667e-3) 
    return alpha_eq
# calculate qli during equilibrium fractionation, based on equation 66 from Wei' 2018 
cdef double q_li_equilibrium_fractionation(double q_ti, double qv, double ql, double alpha_eq_vl) nogil:
    cdef double q_li
    q_li = q_ti/(1+(qv/ql)*alpha_eq_vl)
    return q_li
