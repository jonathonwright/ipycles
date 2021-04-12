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
include "parameters.pxi"

import cython

cdef class IsotopeTracersNone:

    cpdef initialize(self, namelist, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref,
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class IsotopeTracers_NoMicrophysics:
    cdef:
        double T_surface
        double qt_surface
        double p_surface

    cpdef initialize(namelist, self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, ReferenceState.ReferenceState Ref,
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   TimeStepping.TimeStepping TS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef double equilibrium_fractionation_factor(double t) nogil
cdef double q_li_equilibrium_fractionation(double q_ti, double qv, double qc, double alpha_eq_vl) nogil