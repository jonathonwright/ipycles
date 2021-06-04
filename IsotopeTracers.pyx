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

include 'parameters.pxi'

cdef extern from "isotope.h":
    void statsIO_isotope_scaling_magnitude(Grid.DimStruct *dims, double *tmp_values) nogil
    void iso_equilibrium_fractionation_No_Microphysics(Grid.DimStruct *dims, int water_type, double *t, double *qt, double *qv, 
        double *ql, double *qt_iso, double *qv_iso, double *ql_iso) nogil
    void delta_isotopologue(Grid.DimStruct *dims, double *qt, double *qv, double *ql, 
        double *qt_iso, double *qv_iso, double *ql_iso, double *delta_qt, double *delta_qv, double *delta_ql) nogil

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
    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with isotopenone')
        return
    cpdef update(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

cdef class IsotopeTracers_NoMicrophysics:
    def __init__(self, namelist):
        return
    cpdef initialize(self, namelist, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, 
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        Pa.root_print('initialized with IsotopeTracer with No Microphysics')
        # Prognostic variable: standerd water tracer of qt, ql and qv, which are totally same as qt, ql and qv 
        PV.add_variable('qt_tracer', 'kg/kg','qt_tracer','Total water tracer specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_tracer', 'kg/kg','qv_tracer','Vapor water tracer specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_tracer', 'kg/kg','ql_tracer','Cloud liquid water tracer specific humidity','sym', 'scalar', Pa)
        
        # Prognostic variable: qt_iso, total water isotopic specific humidity, defined as the ratio of isotopic mass of H2O18 to moist air.
        PV.add_variable('qt_iso', 'kg/kg','qt_isotope','Total water isotopic specific humidity','sym', "scalar", Pa)
        PV.add_variable('qv_iso', 'kg/kg','qv_isotope','Vapor water isotopic specific humidity','sym', 'scalar', Pa)
        PV.add_variable('ql_iso', 'kg/kg','ql_isotope','Cloud liquid water isotopic specific humidity','sym', 'scalar', Pa)

        # DV isotope tracer ql_iso and qv_iso
        DV.add_variables('qv_iso_DV', 'kg/kg', r'qv_isotope', 'Vapor isotope DV', 'sym', Pa)
        DV.add_variables('ql_iso_DV', 'kg/kg', r'ql_isotope', 'Liquid isotope DV', 'sym', Pa)

        # DV delta of (qt_iso, qt), (qv_iso, qv) and (ql_iso, ql), which will be calculated during fractionation
        DV.add_variables('delta_qt', 'permil', 'delta_qt', 'delta of qt', 'sym', Pa)
        DV.add_variables('delta_qv', 'permil', 'delta_qv', 'delta of qv', 'sym', Pa)
        DV.add_variables('delta_ql', 'permil', 'delta_ql', 'delta of ql', 'sym', Pa)
        DV.add_variables('delta_qv_DV', 'permil', 'delta_qv_DV', 'delta of qv when qv_iso is DV', 'sym', Pa)
        DV.add_variables('delta_ql_DV', 'permil', 'delta_ql_DV', 'delta of qv when qv_iso is DV', 'sym', Pa)

        # finial output results after selection and scaling
        NS.add_profile('qt_iso', Gr, Pa, 'kg/kg', '', 'Finial result of total water isotopic specific humidity')
        NS.add_profile('qv_iso', Gr, Pa, 'kg/kg', '', 'Finial result of vapor isotopic specific humidity')
        NS.add_profile('ql_iso', Gr, Pa, 'kg/kg', '', 'Finial result of liquid isotopic sepcific humidity')
        NS.add_profile('delta_qt', Gr, Pa, 'permil', '', 'delta of qt, calculated by qt_iso/qt during fractioantion')
        NS.add_profile('delta_qv', Gr, Pa, 'permil', '', 'delta of qv, calculated by qt_iso/qt during fractioantion')
        return
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr,'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t tracer_type = 0
            Py_ssize_t HDO_type = 1
            Py_ssize_t H2O18_type = 2
        # fractionation of standard water tracer
        cdef: 
            Py_ssize_t qt_tracer_shift = PV.get_varshift(Gr,'qt_tracer')
            Py_ssize_t qv_tracer_shift = PV.get_varshift(Gr,'qv_tracer')
            Py_ssize_t ql_tracer_shift = PV.get_varshift(Gr,'ql_tracer')
        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, tracer_type, &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], 
                &PV.values[qt_tracer_shift], &PV.values[qv_tracer_shift], &PV.values[ql_tracer_shift])

        # fractionation of isotopes when qv and ql are PVs
        cdef:
            Py_ssize_t qt_iso_shift = PV.get_varshift(Gr,'qt_iso')
            Py_ssize_t qv_iso_shift = PV.get_varshift(Gr,'qv_iso')
            Py_ssize_t ql_iso_shift = PV.get_varshift(Gr,'ql_iso')
            Py_ssize_t delta_qt_shift = DV.get_varshift(Gr, 'delta_qt')
            Py_ssize_t delta_qv_shift = DV.get_varshift(Gr, 'delta_qv')
            Py_ssize_t delta_ql_shift = DV.get_varshift(Gr, 'delta_ql')
        
        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, H2O18_type, &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], 
                &PV.values[qt_iso_shift], &PV.values[qv_iso_shift], &PV.values[ql_iso_shift])
        delta_isotopologue(&Gr.dims, &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], &PV.values[qt_iso_shift], &PV.values[qv_iso_shift], &PV.values[ql_iso_shift],
                &DV.values[delta_qt_shift], &DV.values[delta_qv_shift], &DV.values[delta_ql_shift]) 

        # fractionation of isotopes when qv and ql are DVs
        cdef:
            Py_ssize_t qv_iso_DV_shift = DV.get_varshift(Gr,'qv_iso_DV')
            Py_ssize_t ql_iso_DV_shift = DV.get_varshift(Gr,'ql_iso_DV')
            Py_ssize_t delta_qv_DV_shift = DV.get_varshift(Gr,'delta_qv_DV')
            Py_ssize_t delta_ql_DV_shift = DV.get_varshift(Gr,'delta_ql_DV')
            double [:] tmp_delta = np.zeros(Gr.dims.npg, dtype = np.double, order='c')
        iso_equilibrium_fractionation_No_Microphysics(&Gr.dims, H2O18_type, &DV.values[t_shift], &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], 
                &PV.values[qt_iso_shift], &DV.values[qv_iso_DV_shift], &DV.values[ql_iso_DV_shift])
        delta_isotopologue(&Gr.dims, &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[ql_shift], &PV.values[qt_iso_shift], &DV.values[qv_iso_DV_shift], &DV.values[ql_iso_DV_shift],
                &tmp_delta[0], &DV.values[delta_qv_DV_shift], &DV.values[delta_ql_DV_shift])  


    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            double [:] tmp = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
            Py_ssize_t qt_iso_shift = PV.get_varshift(Gr,'qt_iso')
            Py_ssize_t ql_iso_DV_shift = DV.get_varshift(Gr,'ql_iso_DV')
            Py_ssize_t qv_iso_DV_shift = DV.get_varshift(Gr,'qv_iso_DV')
            Py_ssize_t delta_qt_shift = DV.get_varshift(Gr,'delta_qt')
            Py_ssize_t delta_qv_shift = DV.get_varshift(Gr,'delta_qv')

        tmp = Pa.HorizontalMean(Gr, &PV.values[qt_iso_shift])
        statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
        NS.write_profile('qt_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &DV.values[qv_iso_DV_shift])
        statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
        NS.write_profile('qv_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &DV.values[ql_iso_DV_shift])
        statsIO_isotope_scaling_magnitude(&Gr.dims, &tmp[0]) # scaling back to correct magnitude
        NS.write_profile('ql_iso', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp = Pa.HorizontalMean(Gr, &DV.values[delta_qt_shift]) # scaling back to correct magnitude
        NS.write_profile('delta_qt', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
        
        tmp = Pa.HorizontalMean(Gr, &DV.values[delta_qv_shift]) # scaling back to correct magnitude
        NS.write_profile('delta_qv', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)   
        return
        