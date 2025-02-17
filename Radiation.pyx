#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport TimeStepping
cimport Surface
from Forcing cimport AdjustedMoistAdiabat
from Thermodynamics cimport LatentHeat
# import pylab as plt

import numpy as np
cimport numpy as np
import netCDF4 as nc
from scipy.interpolate import pchip_interpolate
from libc.math cimport pow, cbrt, exp, fmin, fmax, cos, sqrt, sin
from thermodynamic_functions cimport cpm_c
include 'parameters.pxi'
from profiles import profile_data
# import matplotlib.pyplot as plt
import sys


def RadiationFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
    # if namelist specifies RRTM is to be used, this will override any case-specific radiation schemes
    try:
        use_rrtm = namelist['radiation']['use_RRTM']
    except:
        use_rrtm = False
    if use_rrtm:
        return RadiationRRTM(namelist,LH, Pa)
    else:
        casename = namelist['meta']['casename']
        if casename == 'DYCOMS_RF01':
            return RadiationDyCOMS_RF01(namelist)
        elif casename == 'DYCOMS_RF02':
            #Dycoms RF01 and RF02 use the same radiation
            return RadiationDyCOMS_RF01(namelist)
        elif casename == 'SMOKE':
            return RadiationSmoke()
        elif casename == 'Isdac':
            return RadiationIsdac()
        elif casename == 'IsdacCC':
            return RadiationRRTM(namelist,LH, Pa)
        elif casename == 'CGILS':
            return RadiationRRTM(namelist,LH, Pa)
        elif casename == 'ZGILS':
            return RadiationRRTM(namelist, LH, Pa)
        else:
            return RadiationNone()



cdef class RadiationBase:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.z_pencil = ParallelMPI.Pencil()
        self.z_pencil.initialize(Gr, Pa, 2)
        self.heating_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.dTdt_rad = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        self.uflux_lw = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.dflux_lw = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.uflux_sw = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.dflux_sw = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        self.heating_rate_clear = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.uflux_lw_clear = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.dflux_lw_clear = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.uflux_sw_clear = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.dflux_sw_clear = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')


        NS.add_profile('radiative_heating_rate', Gr, Pa)
        NS.add_profile('radiative_entropy_tendency', Gr, Pa)
        NS.add_profile('radiative_temperature_tendency',Gr, Pa)
        NS.add_profile('lw_flux_up', Gr, Pa)
        NS.add_profile('lw_flux_down', Gr, Pa)
        NS.add_profile('sw_flux_up', Gr, Pa)
        NS.add_profile('sw_flux_down', Gr,Pa)
        NS.add_ts('srf_lw_flux_up', Gr, Pa)
        NS.add_ts('srf_lw_flux_down', Gr, Pa)
        NS.add_ts('srf_sw_flux_up', Gr, Pa)
        NS.add_ts('srf_sw_flux_down', Gr, Pa)
        NS.add_ts('toa_lw_flux_up', Gr, Pa)
        NS.add_ts('toa_lw_flux_down', Gr, Pa)
        NS.add_ts('toa_sw_flux_up', Gr, Pa)
        NS.add_ts('toa_sw_flux_down', Gr, Pa)

        NS.add_profile('radiative_heating_rate_clear', Gr, Pa)
        NS.add_profile('lw_flux_up_clear', Gr, Pa)
        NS.add_profile('lw_flux_down_clear', Gr, Pa)
        NS.add_profile('sw_flux_up_clear', Gr, Pa)
        NS.add_profile('sw_flux_down_clear', Gr,Pa)
        NS.add_ts('srf_lw_flux_up_clear', Gr, Pa)
        NS.add_ts('srf_lw_flux_down_clear', Gr, Pa)
        NS.add_ts('srf_sw_flux_up_clear', Gr, Pa)
        NS.add_ts('srf_sw_flux_down_clear', Gr, Pa)
        NS.add_ts('toa_lw_flux_up_clear', Gr, Pa)
        NS.add_ts('toa_lw_flux_down_clear', Gr, Pa)
        NS.add_ts('toa_sw_flux_up_clear', Gr, Pa)
        NS.add_ts('toa_sw_flux_down_clear', Gr, Pa)


        return

    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef stats_io(self, Grid.Grid Gr,  ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift, jshift, ijk

            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] entropy_tendency = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] tmp

        # Now update entropy tendencies
        with nogil:
            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        entropy_tendency[ijk] =  self.heating_rate[ijk] * Ref.alpha0_half[k] / DV.values[ijk + t_shift]

        tmp = Pa.HorizontalMean(Gr, &self.heating_rate[0])
        NS.write_profile('radiative_heating_rate', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &entropy_tendency[0])
        NS.write_profile('radiative_entropy_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.dTdt_rad[0])
        NS.write_profile('radiative_temperature_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.uflux_lw[0])
        NS.write_profile('lw_flux_up', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.dflux_lw[0])
        NS.write_profile('lw_flux_down', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.uflux_sw[0])
        NS.write_profile('sw_flux_up', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.dflux_sw[0])
        NS.write_profile('sw_flux_down', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_ts('srf_lw_flux_up',self.srf_lw_up, Pa ) # Units are W/m^2
        NS.write_ts('srf_lw_flux_down', self.srf_lw_down, Pa)
        NS.write_ts('srf_sw_flux_up', self.srf_sw_up, Pa)
        NS.write_ts('srf_sw_flux_down', self.srf_sw_down, Pa)

        NS.write_ts('toa_lw_flux_up',self.toa_lw_up, Pa ) # Units are W/m^2
        NS.write_ts('toa_lw_flux_down', self.toa_lw_down, Pa)
        NS.write_ts('toa_sw_flux_up', self.toa_sw_up, Pa)
        NS.write_ts('toa_sw_flux_down', self.toa_sw_down, Pa)



        ###
        tmp = Pa.HorizontalMean(Gr, &self.heating_rate_clear[0])
        NS.write_profile('radiative_heating_rate_clear', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.uflux_lw_clear[0])
        NS.write_profile('lw_flux_up_clear', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.dflux_lw_clear[0])
        NS.write_profile('lw_flux_down_clear', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.uflux_sw_clear[0])
        NS.write_profile('sw_flux_up_clear', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.dflux_sw_clear[0])
        NS.write_profile('sw_flux_down_clear', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_ts('srf_lw_flux_up_clear',self.srf_lw_up_clear, Pa ) # Units are W/m^2
        NS.write_ts('srf_lw_flux_down_clear', self.srf_lw_down_clear, Pa)
        NS.write_ts('srf_sw_flux_up_clear', self.srf_sw_up_clear, Pa)
        NS.write_ts('srf_sw_flux_down_clear', self.srf_sw_down_clear, Pa)

        NS.write_ts('toa_lw_flux_up_clear',self.toa_lw_up_clear, Pa ) # Units are W/m^2
        NS.write_ts('toa_lw_flux_down_clear', self.toa_lw_down_clear, Pa)
        NS.write_ts('toa_sw_flux_up_clear', self.toa_sw_up_clear, Pa)
        NS.write_ts('toa_sw_flux_down_clear', self.toa_sw_down_clear, Pa)
        return


cdef class RadiationNone(RadiationBase):
    def __init__(self):
        return
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur,TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class RadiationDyCOMS_RF01(RadiationBase):
    def __init__(self, namelist):
        self.alpha_z = 1.0
        self.kap = 85.0
        try:
            self.f0 = namelist['radiation']['dycoms_f0']
        except:
            self.f0 = 70.0
        self.f1 = 22.0
        self.divergence = 3.75e-6

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.initialize(self, Gr, NS, Pa)

        return

    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur,TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] ql_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[ql_shift])
            double [:, :] qt_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &PV.values[qt_shift])
            double[:, :] f_rad = np.empty((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double[:, :] f_heat = np.empty((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')
            double q_0
            double q_1

            double zi
            double rhoi
            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.z
            double[:] rho = Ref.rho0
            double[:] rho_half = Ref.rho0_half
            double cbrt_z = 0

        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):

                # Compute zi (level of 8.0 g/kg isoline of qt)
                for k in xrange(Gr.dims.n[2]):
                    if qt_pencils[pi, k] > 8e-3:
                        zi = z[gw + k]
                        rhoi = rho_half[gw + k]

                # Now compute the third term on RHS of Stevens et al 2005
                # (equation 3)
                f_rad[pi, 0] = 0.0
                for k in xrange(Gr.dims.n[2]):
                    if z[gw + k] >= zi:
                        cbrt_z = cbrt(z[gw + k] - zi)
                        f_rad[pi, k + 1] = rhoi * cpd * self.divergence * self.alpha_z * (pow(cbrt_z,4)  / 4.0
                                                                                     + zi * cbrt_z)
                    else:
                        f_rad[pi, k + 1] = 0.0

                # Compute the second term on RHS of Stevens et al. 2005
                # (equation 3)
                q_1 = 0.0
                f_rad[pi, 0] += self.f1 * exp(-q_1)
                for k in xrange(1, Gr.dims.n[2] + 1):
                    q_1 += self.kap * \
                        rho_half[gw + k - 1] * ql_pencils[pi, k - 1] * dz
                    f_rad[pi, k] += self.f1 * exp(-q_1)

                # Compute the first term on RHS of Stevens et al. 2005
                # (equation 3)
                q_0 = 0.0
                f_rad[pi, Gr.dims.n[2]] += self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * ql_pencils[pi, k] * dz
                    f_rad[pi, k] += self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[pi, k] = - \
                       (f_rad[pi, k + 1] - f_rad[pi, k]) * dzi / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &self.heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[ijk] / DV.values[ijk + t_shift] 
                        self.dTdt_rad[ijk] = self.heating_rate[ijk] / cpm_c(PV.values[ijk + qt_shift]) 

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.stats_io(self, Gr, Ref, DV, NS, Pa)


        return


cdef class RadiationSmoke(RadiationBase):
    '''
    Radiation for the smoke cloud case

    Bretherton, C. S., and coauthors, 1999:
    An intercomparison of radiatively- driven entrainment and turbulence in a smoke cloud,
    as simulated by different numerical models. Quart. J. Roy. Meteor. Soc., 125, 391-423. Full text copy.

    '''


    def __init__(self):
        self.f0 = 60.0
        self.kap = 0.02

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.initialize(self, Gr, NS, Pa)
        return
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t smoke_shift = PV.get_varshift(Gr, 'smoke')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] smoke_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &PV.values[smoke_shift])
            double[:, :] f_rad = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double[:, :] f_heat = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')

            double q_0

            double zi
            double rhoi
            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.z
            double[:] rho = Ref.rho0
            double[:] rho_half = Ref.rho0_half
            double cbrt_z = 0
            Py_ssize_t kk


        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):

                q_0 = 0.0
                f_rad[pi, Gr.dims.n[2]] = self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * smoke_pencils[pi, k] * dz
                    f_rad[pi, k] = self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[pi, k] = - \
                       (f_rad[pi, k + 1] - f_rad[pi, k]) * dzi / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &self.heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[ijk] / DV.values[ijk + t_shift] * Ref.alpha0_half[k]
                        self.dTdt_rad[ijk] = self.heating_rate[ijk] / cpd * Ref.alpha0_half[k]

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.stats_io(self, Gr, Ref, DV, NS, Pa)

        return


cdef class RadiationIsdac(RadiationBase):
    def __init__(self):
        self.kap = 170.0
        self.f0 = 72.0
        self.f1 = 15.0

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.initialize(self, Gr, NS, Pa)

        self.radiative_flux = np.zeros(Gr.dims.npg, dtype=np.double, order='c')
        NS.add_profile('radiative_flux',Gr, Pa)
        return

    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] ql_pencils =  self.z_pencil.forward_double(& Gr.dims, Pa, & DV.values[ql_shift])
            double [:, :] qt_pencils =  self.z_pencil.forward_double(& Gr.dims, Pa, & PV.values[qt_shift])
            double [:, :] f_rad = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double [:, :] f_heat = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')
            double [:] heating_rate = self.heating_rate[:]
            double [:] radiative_flux = self.radiative_flux[:]
            double q_0
            double q_1

            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.z
            double[:] rho = Ref.rho0
            double[:] rho_half = Ref.rho0_half

        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):

                # Compute the second term on RHS of Stevens et al. 2005
                # (equation 3)
                q_1 = 0.0
                f_rad[pi, 0] += self.f1 * exp(-q_1)
                for k in xrange(1, Gr.dims.n[2] + 1):
                    q_1 += self.kap * \
                        rho_half[gw + k - 1] * ql_pencils[pi, k - 1] * dz
                    f_rad[pi, k] += self.f1 * exp(-q_1)

                # Compute the first term on RHS of Stevens et al. 2005
                # (equation 3)
                q_0 = 0.0
                f_rad[pi, Gr.dims.n[2]] += self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * ql_pencils[pi, k] * dz
                    f_rad[pi, k] += self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[pi, k] = - \
                        (f_rad[pi, k + 1] - f_rad[pi, k]) * dzi / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(& Gr.dims, Pa, f_heat, & heating_rate[0])
        self.z_pencil.reverse_double(& Gr.dims, Pa, f_rad[:,:-1], & radiative_flux[0])



        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] += heating_rate[ijk] / DV.values[ijk + t_shift]
                        self.dTdt_rad[ijk] = self.heating_rate[ijk] / cpm_c(PV.values[ijk + qt_shift])

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.stats_io(self, Gr, Ref, DV, NS, Pa)

        cdef double [:] tmp

        tmp = Pa.HorizontalMean(Gr, &self.radiative_flux[0])
        NS.write_profile('radiative_flux',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

        return


# Note: the RRTM modules are compiled in the 'RRTMG' directory:
cdef extern:
    void c_rrtmg_lw_init(double *cpdair)
    void c_rrtmg_lw (
             int *ncol    ,int *nlay    ,int *icld    ,int *idrv    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *cfc11vmr,double *cfc12vmr,double *cfc22vmr,double *ccl4vmr ,double *emis    ,
             int *inflglw ,int *iceflglw,int *liqflglw,double *cldfr   ,
             double *taucld  ,double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,
             double *uflx    ,double *dflx    ,double *hr      ,double *uflxc   ,double *dflxc,  double *hrc,
             double *duflx_dt,double *duflxc_dt )
    void c_rrtmg_sw_init(double *cpdair)
    void c_rrtmg_sw (int *ncol    ,int *nlay    ,int *icld    ,int *iaer    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *asdir   ,double *asdif   ,double *aldir   ,double *aldif   ,
             double *coszen  ,double *adjes   ,int *dyofyr  ,double *scon    ,
             int *inflgsw ,int *iceflgsw,int *liqflgsw,double *cldfr   ,
             double *taucld  ,double *ssacld  ,double *asmcld  ,double *fsfcld  ,
             double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,double *ssaaer  ,double *asmaer  ,double *ecaer   ,
             double *swuflx  ,double *swdflx  ,double *swhr    ,double *swuflxc ,double *swdflxc ,double *swhrc)




cdef class RadiationRRTM(RadiationBase):

    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):


        # Required for surface energy budget calculations, can also be used for stats io
        self.srf_lw_down = 0.0
        self.srf_sw_down = 0.0
        self.srf_lw_up = 0.0
        self.srf_sw_up = 0.0

        self.toa_lw_down = 0.0
        self.toa_sw_down = 0.0
        self.toa_lw_up = 0.0
        self.toa_sw_up = 0.0
        casename = namelist['meta']['casename']
        self.modified_adiabat = False

        if casename == 'Sheba':
            self.profile_name = 'arctic'
        elif casename == 'Isdac':
            self.profile_name = 'arctic'
        elif casename == 'IsdacCC':
            self.profile_name = 'arctic'
        elif casename == 'Mpace':
            self.profile_name = 'arctic'
        elif casename == 'DYCOMS_RF01':
            self.profile_name = 'cgils_ctl_s12'
        elif casename == 'CGILS':
            loc = namelist['meta']['CGILS']['location']
            is_p2 = namelist['meta']['CGILS']['P2']
            if is_p2:
                self.profile_name = 'cgils_p2_s'+str(loc)
            else:
                self.profile_name = 'cgils_ctl_s'+str(loc)
        elif casename == 'ZGILS':
            loc = namelist['meta']['ZGILS']['location']
            self.profile_name = 'cgils_ctl_s'+str(loc)
            self.modified_adiabat = True
            self.reference_profile = AdjustedMoistAdiabat(namelist, LH, Pa)
            self.Tg_adiabat = 295.0
            self.Pg_adiabat = 1000.0e2
            self.RH_adiabat = 0.3

        else:
            Pa.root_print('RadiationRRTM: Case ' + casename + ' has no known extension profile')
            Pa.kill()

        # Namelist options related to the profile extension
        try:
            self.n_buffer = namelist['radiation']['RRTM']['buffer_points']
        except:
            self.n_buffer = 0
        try:
            self.stretch_factor = namelist['radiation']['RRTM']['stretch_factor']
        except:
            self.stretch_factor = 1.0

        try:
            self.patch_pressure = namelist['radiation']['RRTM']['patch_pressure']
        except:
            self.patch_pressure = 1000.00*100.0

        # Namelist options related to gas concentrations
        try:
            self.co2_factor = namelist['radiation']['RRTM']['co2_factor']
        except:
            self.co2_factor = 1.0

        try:
            self.h2o_factor = namelist['radiation']['RRTM']['h2o_factor']
        except:
            self.h2o_factor = 1.0

        # Namelist options related to insolation

        try:
            self.dyofyr_init = namelist['radiation']['RRTM']['dyofyr']
        except:
            self.dyofyr_init = 0
        self.dyofyr = self.dyofyr_init

        #Adjes is not used if dyofyr > 0
        try:
            self.adjes = namelist['radiation']['RRTM']['adjes']
        except:
            Pa.root_print('Insolation adjustive factor not set so RadiationRRTM takes default value: adjes = 0.5 (12 hour of daylight).')
            self.adjes = 0.5

        try:
            self.daily_mean_sw = namelist['radiation']['RRTM']['daily_mean_sw']
        except:
            print('Use daily mean sw in RRTM!')
            self.daily_mean_sw = True

        if self.daily_mean_sw:
            try:
                self.coszen = namelist['radiation']['RRTM']['coszen']
            except:
                Pa.root_print('Mean Daytime cos(SZA) not set so RadiationRRTM takes default value: coszen = 2.0/pi .')
                self.coszen = 2.0/pi
        else:
            try:
                self.latitude = namelist['radiation']['RRTM']['latitude']
                self.longitude = namelist['radiation']['RRTM']['longitude']
                self.hourz_init = namelist['radiation']['RRTM']['hourz']
                self.coszen = cos_sza(self.dyofyr_init, self.hourz_init, self.latitude, self.longitude)
                print('Calculated cos(sza) based on time and location, cos(sza) = %5.4f .'%(self.coszen))
            except:
                print('Hourz, latitude, or longitude not set. Cannot calculate RRTM SW. Exit now!')
                sys.exit()

        try:
            self.scon = namelist['radiation']['RRTM']['solar_constant']
        except:
            Pa.root_print('Solar Constant not set so RadiationRRTM takes default value: scon = 1360.0 .')
            self.scon = 1360.0

        try:
            self.adif = namelist['radiation']['RRTM']['adif']
        except:
            Pa.root_print('Surface diffusive albedo not set so RadiationRRTM takes default value: adif = 0.06 .')
            self.adif = 0.06

        try:
            self.adir = namelist['radiation']['RRTM']['adir']
        except:
            if (self.coszen > 0.0):
                self.adir = (.026/(self.coszen**1.7 + .065)+(.15*(self.coszen-0.10)*(self.coszen-0.50)*(self.coszen- 1.00)))
            else:
                self.adir = 0.0
            Pa.root_print('Surface direct albedo not set so RadiationRRTM computes value: adif = %5.4f .'%(self.adir))

        try:
            self.uniform_reliq = namelist['radiation']['RRTM']['uniform_reliq']
        except:
            Pa.root_print('uniform_reliq not set so RadiationRRTM takes default value: uniform_reliq = False.')
            self.uniform_reliq = False

        try:
            self.radiation_frequency = namelist['radiation']['RRTM']['frequency']
        except:
            Pa.root_print('radiation_frequency not set so RadiationRRTM takes default value: radiation_frequency = 0.0 (compute at every step).')
            self.radiation_frequency = 0.0



        self.next_radiation_calculate = 0.0

        #For IsdacCC simulations, initial profiles need to be justified correspondingly
        try:
            self.IsdacCC_dT = namelist['initial']['dSST'] + namelist['initial']['dTi'] - 5.0
            print('IsdacCC case: RRTM profiles are shifted according to %2.2f temperature change.'%(self.IsdacCC_dT))
        except:
            self.IsdacCC_dT = 0.0

        return


    cpdef initialize(self, Grid.Grid Gr,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.initialize(self, Gr, NS, Pa)
        #Initialize rrtmg_lw and rrtmg_sw
        cdef double cpdair = np.float64(cpd)
        c_rrtmg_lw_init(&cpdair)
        c_rrtmg_sw_init(&cpdair)

        cdef:
            Py_ssize_t nz = Gr.dims.n[2]
            Py_ssize_t n_pencils = self.z_pencil.n_local_pencils

        self.uflx_lw_pencils = np.zeros((n_pencils, nz), dtype=np.double, order='c')
        self.dflx_lw_pencils = np.zeros((n_pencils, nz), dtype=np.double, order='c')
        self.uflx_sw_pencils = np.zeros((n_pencils, nz), dtype=np.double, order='c')
        self.dflx_sw_pencils = np.zeros((n_pencils, nz), dtype=np.double, order='c')

        return



    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                              Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):


        cdef:
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:,:] qv_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[qv_shift])
            double [:,:] t_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[t_shift])
            Py_ssize_t nz = Gr.dims.n[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t i,k
            Py_ssize_t n_adiabat
            double [:] pressures_adiabat


        # Construct the extension of the profiles, including a blending region between the given profile and LES domain (if desired)
        if self.modified_adiabat:
            # pressures = profile_data[self.profile_name]['pressure'][:]
            pressures = np.arange(25*100, 1015*100, 10*100)
            pressures = np.array(pressures[::-1], dtype=np.double)
            n_adiabat = np.shape(pressures)[0]
            self.reference_profile.initialize(Pa, pressures, n_adiabat, self.Pg_adiabat, self.Tg_adiabat, self.RH_adiabat)
            temperatures =np.array( self.reference_profile.temperature)
            vapor_mixing_ratios = np.array(self.reference_profile.rv)

        else:
            pressures = profile_data[self.profile_name]['pressure'][:]
            temperatures = profile_data[self.profile_name]['temperature'][:]
            vapor_mixing_ratios = profile_data[self.profile_name]['vapor_mixing_ratio'][:]

        # Sanity check that patch_pressure < minimum LES domain pressure
        dp = np.abs(Ref.p0_half_global[nz + gw -1] - Ref.p0_half_global[nz + gw -2])
        self.patch_pressure = np.minimum(self.patch_pressure, Ref.p0_half_global[nz + gw -1] - dp  )

        #n_profile = len(pressures[pressures<=self.patch_pressure]) # nprofile = # of points in the fixed profile to use
        # above syntax tends to cause problems so use a more robust way
        n_profile = 0
        for pressure in pressures:
            if pressure <= self.patch_pressure:
                n_profile += 1

        self.n_ext =  n_profile + self.n_buffer # n_ext = total # of points to add to LES domain (buffer portion + fixed profile portion)


        # Create the space for the extensions (to be tacked on to top of LES pencils)
        # we declare these as class members in case we want to modify the buffer zone during run time
        # i.e. if there is some drift to top of LES profiles

        self.p_ext = np.zeros((self.n_ext,),dtype=np.double)
        self.t_ext = np.zeros((self.n_ext,),dtype=np.double)
        self.rv_ext = np.zeros((self.n_ext,),dtype=np.double)
        cdef Py_ssize_t count = 0
        if self.IsdacCC_dT == 0.0:
            for k in xrange(len(pressures)-n_profile, len(pressures)):
                self.p_ext[self.n_buffer+count] = pressures[k]
                self.t_ext[self.n_buffer+count] = temperatures[k]
                self.rv_ext[self.n_buffer+count] = vapor_mixing_ratios[k]
                count += 1
        else:
            specific_humidity = vapor_mixing_ratios / (1.0 + vapor_mixing_ratios)
            for k in xrange(len(pressures)-n_profile, len(pressures)):
                self.p_ext[self.n_buffer+count] = pressures[k]
                qt_new = get_humidity(temperatures[k], specific_humidity[k], pressures[k], temperatures[k]+self.IsdacCC_dT, Th)
                self.t_ext[self.n_buffer+count] = temperatures[k] + self.IsdacCC_dT
                self.rv_ext[self.n_buffer+count] = qt_new / (1.0 - qt_new)
                count += 1


        # Now  create the buffer zone
        if self.n_buffer > 0:
            dp = np.abs(Ref.p0_half_global[nz + gw -1] - Ref.p0_half_global[nz + gw -2])
            self.p_ext[0] = Ref.p0_half_global[nz + gw -1] - dp
            for i in range(1,self.n_buffer):
                self.p_ext[i] = self.p_ext[i-1] - (i+1.0)**self.stretch_factor * dp


            # Sanity check the buffer zone
            if self.p_ext[self.n_buffer-1] < self.p_ext[self.n_buffer]:
                Pa.root_print('Radiation buffer zone extends too far')
                Pa.kill()

            # Pressures of "data" points for interpolation, must be INCREASING pressure
            xi = np.array([self.p_ext[self.n_buffer+1],self.p_ext[self.n_buffer],Ref.p0_half_global[nz + gw -1],Ref.p0_half_global[nz + gw -2] ],dtype=np.double)

            # interpolation for temperature
            ti = np.array([self.t_ext[self.n_buffer+1],self.t_ext[self.n_buffer], t_pencils[0,nz-1],t_pencils[0,nz-2] ], dtype = np.double)
            # interpolation for vapor mixing ratio
            rv_m2 = qv_pencils[0, nz-2]/ (1.0 - qv_pencils[0, nz-2])
            rv_m1 = qv_pencils[0,nz-1]/(1.0-qv_pencils[0,nz-1])
            ri = np.array([self.rv_ext[self.n_buffer+1],self.rv_ext[self.n_buffer], rv_m1, rv_m2 ], dtype = np.double)

            for i in xrange(self.n_buffer):
                self.rv_ext[i] = pchip_interpolate(xi, ri, self.p_ext[i] )
                self.t_ext[i] = pchip_interpolate(xi,ti, self.p_ext[i])

        #--- Plotting to evaluate implementation of buffer zone
        #--- Comment out when not running locally
        for i in xrange(Gr.dims.nlg[2]):
            qv_pencils[0,i] = qv_pencils[0, i]/ (1.0 - qv_pencils[0, i])
        #
        # Plotting to evaluate implementation of buffer zone
        # plt.figure(1)
        # plt.plot(self.rv_ext,self.p_ext,'or')
        # plt.plot(vapor_mixing_ratios, pressures)
        # plt.plot(qv_pencils[0,:], Ref.p0_half_global[gw:-gw],'ob')
        # plt.gca().invert_yaxis()
        # plt.figure(2)
        # plt.plot(self.t_ext,self.p_ext,'-or')
        # plt.plot(temperatures,pressures)
        # plt.plot(t_pencils[0,:], Ref.p0_half_global[gw:-gw],'-ob')
        # plt.gca().invert_yaxis()
        # plt.show()
        #---END Plotting to evaluate implementation of buffer zone


        self.p_full = np.zeros((self.n_ext+nz,), dtype=np.double)
        self.pi_full = np.zeros((self.n_ext+1+nz,),dtype=np.double)

        self.p_full[0:nz] = Ref.p0_half_global[gw:nz+gw]
        self.p_full[nz:]=self.p_ext[:]

        self.pi_full[0:nz] = Ref.p0_global[gw:nz+gw]
        for i in range(nz,self.n_ext+nz):
            self.pi_full[i] = (self.p_full[i] + self.p_full[i-1]) * 0.5
        self.pi_full[self.n_ext +  nz] = 2.0 * self.p_full[self.n_ext + nz -1 ] - self.pi_full[self.n_ext + nz -1]

        # try to get ozone
        try:
            o3_trace = profile_data[self.profile_name]['o3_vmr'][:]   # O3 VMR (from SRF to TOP)
            o3_pressure = profile_data[self.profile_name]['pressure'][:]/100.0       # Pressure (from SRF to TOP) in hPa
            # can't do simple interpolation... Need to conserve column path !!!
            use_o3in = True
        except:
            try:
                o3_trace = profile_data[self.profile_name]['o3_mmr'][:]*28.97/47.9982   # O3 MR converted to VMR
                o3_pressure = profile_data[self.profile_name]['pressure'][:]/100.0       # Pressure (from SRF to TOP) in hPa
                # can't do simple interpolation... Need to conserve column path !!!
                use_o3in = True

            except:
                Pa.root_print('O3 profile not set so default RRTM profile will be used.')
                use_o3in = False


        # Read in trace gas data
        lw_input_file = './RRTMG/lw/data/rrtmg_lw.nc'
        lw_gas = nc.Dataset(lw_input_file,  "r")

        lw_pressure = np.asarray(lw_gas.variables['Pressure'])
        lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
        lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
        lw_ngas = lw_absorber.shape[1]
        lw_np = lw_absorber.shape[0]

        # 9 Gases: O3, CO2, CH4, N2O, O2, CFC11, CFC12, CFC22, CCL4
        # From rad_driver.f90, lines 546 to 552
        trace = np.zeros((9,lw_np),dtype=np.double,order='F')
        for i in xrange(lw_ngas):
            gas_name = ''.join(str(lw_gas.variables['AbsorberNames'][i,:]))
            if 'O3' in gas_name:
                trace[0,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CO2' in gas_name:
                trace[1,:] = lw_absorber[:,i].reshape(1,lw_np)*self.co2_factor
            elif 'CH4' in gas_name:
                trace[2,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'N2O' in gas_name:
                trace[3,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'O2' in gas_name:
                trace[4,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC11' in gas_name:
                trace[5,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC12' in gas_name:
                trace[6,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC22' in gas_name:
                trace[7,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CCL4' in gas_name:
                trace[8,:] = lw_absorber[:,i].reshape(1,lw_np)

        # From rad_driver.f90, lines 585 to 620
        trpath = np.zeros((nz + self.n_ext + 1, 9),dtype=np.double,order='F')
        # plev = self.pi_full[:]/100.0
        for i in xrange(1, nz + self.n_ext + 1):
            trpath[i,:] = trpath[i-1,:]
            if (self.pi_full[i-1]/100.0 > lw_pressure[0]):
                trpath[i,:] = trpath[i,:] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,lw_pressure[0])))/g*trace[:,0]
            for m in xrange(1,lw_np):
                plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m-1]))))
                pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m]))))
                if (plow > pupp):
                    pmid = 0.5*(plow+pupp)
                    wgtlow = (pmid-lw_pressure[m])/(lw_pressure[m-1]-lw_pressure[m])
                    wgtupp = (lw_pressure[m-1]-pmid)/(lw_pressure[m-1]-lw_pressure[m])
                    trpath[i,:] = trpath[i,:] + (plow-pupp)/g*(wgtlow*trace[:,m-1]  + wgtupp*trace[:,m])
            if (self.pi_full[i]/100.0 < lw_pressure[lw_np-1]):
                trpath[i,:] = trpath[i,:] + (np.min((self.pi_full[i-1]/100.0,lw_pressure[lw_np-1]))-self.pi_full[i]/100.0)/g*trace[:,lw_np-1]

        tmpTrace = np.zeros((nz + self.n_ext,9),dtype=np.double,order='F')
        for i in xrange(9):
            for k in xrange(nz + self.n_ext):
                tmpTrace[k,i] = g*100.0/(self.pi_full[k]-self.pi_full[k+1])*(trpath[k+1,i]-trpath[k,i])

        if use_o3in == False:
            self.o3vmr  = np.array(tmpTrace[:,0],dtype=np.double, order='F')
        else:
            # o3_trace, o3_pressure
            trpath_o3 = np.zeros(nz + self.n_ext+1, dtype=np.double, order='F')
            # plev = self.pi_full/100.0
            o3_np = o3_trace.shape[0]
            for i in xrange(1, nz + self.n_ext+1):
                trpath_o3[i] = trpath_o3[i-1]
                if (self.pi_full[i-1]/100.0 > o3_pressure[0]):
                    trpath_o3[i] = trpath_o3[i] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,o3_pressure[0])))/g*o3_trace[0]
                for m in xrange(1,o3_np):
                    plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, o3_pressure[m-1]))))
                    pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, o3_pressure[m]))))
                    if (plow > pupp):
                        pmid = 0.5*(plow+pupp)
                        wgtlow = (pmid-o3_pressure[m])/(o3_pressure[m-1]-o3_pressure[m])
                        wgtupp = (o3_pressure[m-1]-pmid)/(o3_pressure[m-1]-o3_pressure[m])
                        trpath_o3[i] = trpath_o3[i] + (plow-pupp)/g*(wgtlow*o3_trace[m-1]  + wgtupp*o3_trace[m])
                if (self.pi_full[i]/100.0 < o3_pressure[o3_np-1]):
                    trpath_o3[i] = trpath_o3[i] + (np.min((self.pi_full[i-1]/100.0,o3_pressure[o3_np-1]))-self.pi_full[i]/100.0)/g*o3_trace[o3_np-1]
            tmpTrace_o3 = np.zeros( nz + self.n_ext, dtype=np.double, order='F')
            for k in xrange(nz + self.n_ext):
                tmpTrace_o3[k] = g *100.0/(self.pi_full[k]-self.pi_full[k+1])*(trpath_o3[k+1]-trpath_o3[k])
            self.o3vmr = np.array(tmpTrace_o3[:],dtype=np.double, order='F')

        self.co2vmr = np.array(tmpTrace[:,1],dtype=np.double, order='F')
        self.ch4vmr =  np.array(tmpTrace[:,2],dtype=np.double, order='F')
        self.n2ovmr =  np.array(tmpTrace[:,3],dtype=np.double, order='F')
        self.o2vmr  =  np.array(tmpTrace[:,4],dtype=np.double, order='F')
        self.cfc11vmr =  np.array(tmpTrace[:,5],dtype=np.double, order='F')
        self.cfc12vmr =  np.array(tmpTrace[:,6],dtype=np.double, order='F')
        self.cfc22vmr = np.array( tmpTrace[:,7],dtype=np.double, order='F')
        self.ccl4vmr  =  np.array(tmpTrace[:,8],dtype=np.double, order='F')


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa):

        if TS.rk_step == 0:
            if self.radiation_frequency <= 0.0:
                self.update_RRTM(Gr, Ref, PV, DV, Sur, Pa)
            elif TS.t >= self.next_radiation_calculate:
                if not self.daily_mean_sw:
                    #Update coszen
                    self.dyofyr = np.floor_divide(TS.t, 86400.0) + self.dyofyr_init
                    self.hourz = self.hourz_init + TS.t/3600.0
                    if self.hourz > 24.0:
                        self.hourz = np.remainder(self.hourz, 24.0)
                    self.coszen = cos_sza(self.dyofyr, self.hourz, self.latitude, self.longitude)

                self.update_RRTM(Gr, Ref, PV, DV, Sur, Pa)
                self.next_radiation_calculate = (TS.t//self.radiation_frequency + 1.0) * self.radiation_frequency


        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw


            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k

                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[ijk] / DV.values[ijk + t_shift] * Ref.alpha0_half[k]
                        self.dTdt_rad[ijk] = self.heating_rate[ijk] * Ref.alpha0_half[k]/cpm_c(PV.values[ijk + qt_shift])

        return

    cdef update_RRTM(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                      DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t nz = Gr.dims.n[2]
            Py_ssize_t nz_full = self.n_ext + nz
            Py_ssize_t n_pencils = self.z_pencil.n_local_pencils
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qi_shift
            double [:,:] t_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[t_shift])
            double [:,:] qv_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[qv_shift])
            double [:,:] ql_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[ql_shift])
            double [:,:] qi_pencil = np.zeros((n_pencils,nz),dtype=np.double, order='c')
            double [:,:] rl_full = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            Py_ssize_t k, ip
            bint use_ice = False
            Py_ssize_t gw = Gr.dims.gw


        if 'qi' in DV.name_index:
            qi_shift = DV.get_varshift(Gr, 'qi')
            qi_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[qi_shift])
            # use_ice = True  # testing radiation without ice effect



        # Define input arrays for RRTM
        cdef:
            double [:,:] play_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] plev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:,:] tlay_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] tlev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:] tsfc_in = np.ones((n_pencils),dtype=np.double,order='F') * Sur.T_surface
            double [:,:] h2ovmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] o3vmr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] co2vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] ch4vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] n2ovmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] o2vmr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc11vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc12vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc22vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] ccl4vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] emis_in = np.ones((n_pencils,16),dtype=np.double,order='F') * 0.95
            double [:,:] cldfr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cicewp_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cliqwp_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reice_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reliq_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:] coszen_in = np.ones((n_pencils),dtype=np.double,order='F') *self.coszen
            double [:] asdir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] asdif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:] aldir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] aldif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:,:,:] taucld_lw_in  = np.zeros((16,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] tauaer_lw_in  = np.zeros((n_pencils,nz_full,16),dtype=np.double,order='F')
            double [:,:,:] taucld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] ssacld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] asmcld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] fsfcld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] tauaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] ssaaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] asmaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] ecaer_sw_in  = np.zeros((n_pencils,nz_full,6),dtype=np.double,order='F')

            # Output
            double[:,:] uflx_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflx_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hr_lw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] uflxc_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflxc_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hrc_lw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] duflx_dt_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] duflxc_dt_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] uflx_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflx_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hr_sw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] uflxc_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflxc_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hrc_sw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')

            double rv_to_reff = np.exp(np.log(1.2)**2.0)*10.0*1000.0

        with nogil:
            for k in xrange(nz, nz_full):
                for ip in xrange(n_pencils):
                    tlay_in[ip, k] = self.t_ext[k-nz]
                    h2ovmr_in[ip, k] = self.rv_ext[k-nz] * Rv/Rd * self.h2o_factor
                    # Assuming for now that there is no condensate above LES domain!
            for k in xrange(nz):
                for ip in xrange(n_pencils):
                    tlay_in[ip,k] = t_pencil[ip,k]
                    h2ovmr_in[ip,k] = qv_pencil[ip,k]/ (1.0 - qv_pencil[ip,k])* Rv/Rd * self.h2o_factor
                    rl_full[ip,k] = (ql_pencil[ip,k])/ (1.0 - qv_pencil[ip,k])
                    cliqwp_in[ip,k] = ((ql_pencil[ip,k])/ (1.0 - qv_pencil[ip,k])
                                       *1.0e3*(self.pi_full[k] - self.pi_full[k+1])/g)
                    cicewp_in[ip,k] = ((qi_pencil[ip,k])/ (1.0 - qv_pencil[ip,k])
                                       *1.0e3*(self.pi_full[k] - self.pi_full[k+1])/g)
                    if ql_pencil[ip,k] + qi_pencil[ip,k] > ql_threshold:
                        cldfr_in[ip,k] = 1.0


        with nogil:
            for k in xrange(nz_full):
                for ip in xrange(n_pencils):
                    play_in[ip,k] = self.p_full[k]/100.0
                    o3vmr_in[ip, k] = self.o3vmr[k]
                    co2vmr_in[ip, k] = self.co2vmr[k]
                    ch4vmr_in[ip, k] = self.ch4vmr[k]
                    n2ovmr_in[ip, k] = self.n2ovmr[k]
                    o2vmr_in [ip, k] = self.o2vmr[k]
                    cfc11vmr_in[ip, k] = self.cfc11vmr[k]
                    cfc12vmr_in[ip, k] = self.cfc12vmr[k]
                    cfc22vmr_in[ip, k] = self.cfc22vmr[k]
                    ccl4vmr_in[ip, k] = self.ccl4vmr[k]


                    if self.uniform_reliq:
                        reliq_in[ip, k] = 14.0*cldfr_in[ip,k]
                    else:
                        reliq_in[ip, k] = ((3.0*self.p_full[k]/Rd/tlay_in[ip,k]*rl_full[ip,k]/
                                                    fmax(cldfr_in[ip,k],1.0e-6))/(4.0*pi*1.0e3*100.0))**(1.0/3.0)
                        reliq_in[ip, k] = fmin(fmax(reliq_in[ip, k]*rv_to_reff, 2.5), 60.0)

                    # Boudala et al. (2002) Eqn 10a
                    reice_in[ip, k] = 53.005 * ((self.p_full[k]/Rd/tlay_in[ip, k]*qi_pencil[ip, k]*1.0e3)/
                                                fmax(cldfr_in[ip,k],1.0e-6)) ** 0.06 \
                                      * exp(0.013*(tlay_in[ip, k] - 273.16))
                    reice_in[ip, k] = fmin(fmax(reice_in[ip, k]/1.54, 5.0), 140.0) # Threshold from rrtmg sw instruction

            for ip in xrange(n_pencils):
                tlev_in[ip, 0] = Sur.T_surface
                plev_in[ip,0] = self.pi_full[0]/100.0
                for k in xrange(1,nz_full):
                    tlev_in[ip, k] = 0.5*(tlay_in[ip,k-1]+tlay_in[ip,k])
                    plev_in[ip,k] = self.pi_full[k]/100.0
                tlev_in[ip, nz_full] = 2.0*tlay_in[ip,nz_full-1] - tlev_in[ip,nz_full-1]
                plev_in[ip,nz_full] = self.pi_full[nz_full]/100.0


        cdef:
            int ncol = n_pencils
            int nlay = nz_full
            int icld = 1
            int idrv = 0
            int iaer = 0
            int inflglw = 2
            int iceflglw = 3
            int liqflglw = 1
            int inflgsw = 2
            int iceflgsw = 3
            int liqflgsw = 1

        c_rrtmg_lw (
             &ncol    ,&nlay    ,&icld    ,&idrv,
             &play_in[0,0]    ,&plev_in[0,0]    ,&tlay_in[0,0]    ,&tlev_in[0,0]    ,&tsfc_in[0]    ,
             &h2ovmr_in[0,0]  ,&o3vmr_in[0,0]   ,&co2vmr_in[0,0]  ,&ch4vmr_in[0,0]  ,&n2ovmr_in[0,0]  ,&o2vmr_in[0,0],
             &cfc11vmr_in[0,0],&cfc12vmr_in[0,0],&cfc22vmr_in[0,0],&ccl4vmr_in[0,0] ,&emis_in[0,0]    ,
             &inflglw ,&iceflglw,&liqflglw,&cldfr_in[0,0]   ,
             &taucld_lw_in[0,0,0]  ,&cicewp_in[0,0]  ,&cliqwp_in[0,0]  ,&reice_in[0,0]   ,&reliq_in[0,0]   ,
             &tauaer_lw_in[0,0,0]  ,
             &uflx_lw_out[0,0]    ,&dflx_lw_out[0,0]    ,&hr_lw_out[0,0]      ,&uflxc_lw_out[0,0]   ,&dflxc_lw_out[0,0],  &hrc_lw_out[0,0],
             &duflx_dt_out[0,0],&duflxc_dt_out[0,0] )


        c_rrtmg_sw (
            &ncol, &nlay, &icld, &iaer, &play_in[0,0], &plev_in[0,0], &tlay_in[0,0], &tlev_in[0,0],&tsfc_in[0],
            &h2ovmr_in[0,0], &o3vmr_in[0,0], &co2vmr_in[0,0], &ch4vmr_in[0,0], &n2ovmr_in[0,0],&o2vmr_in[0,0],
             &asdir_in[0]   ,&asdif_in[0]   ,&aldir_in[0]   ,&aldif_in[0]   ,
             &coszen_in[0]  ,&self.adjes   ,&self.dyofyr  ,&self.scon   ,
             &inflgsw ,&iceflgsw,&liqflgsw,&cldfr_in[0,0]   ,
             &taucld_sw_in[0,0,0]  ,&ssacld_sw_in[0,0,0]  ,&asmcld_sw_in[0,0,0]  ,&fsfcld_sw_in[0,0,0]  ,
             &cicewp_in[0,0]  ,&cliqwp_in[0,0]  ,&reice_in[0,0]   ,&reliq_in[0,0]   ,
             &tauaer_sw_in[0,0,0]  ,&ssaaer_sw_in[0,0,0]  ,&asmaer_sw_in[0,0,0]  ,&ecaer_sw_in[0,0,0]   ,
             &uflx_sw_out[0,0]    ,&dflx_sw_out[0,0]    ,&hr_sw_out[0,0]      ,&uflxc_sw_out[0,0]   ,&dflxc_sw_out[0,0], &hrc_sw_out[0,0])


        cdef double [:,:] heating_rate_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] uflux_lw_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] dflux_lw_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] uflux_sw_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] dflux_sw_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')

        cdef double [:,:] heating_rate_clear_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] uflux_lw_clear_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] dflux_lw_clear_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] uflux_sw_clear_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double [:,:] dflux_sw_clear_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')

        cdef double srf_lw_up_local =0.0, srf_lw_down_local=0.0, srf_sw_up_local=0.0, srf_sw_down_local=0.0
        cdef double toa_lw_up_local =0.0, toa_lw_down_local=0.0, toa_sw_up_local=0.0, toa_sw_down_local=0.0

        cdef double srf_lw_up_clear_local =0.0, srf_lw_down_clear_local=0.0, srf_sw_up_clear_local=0.0, srf_sw_down_clear_local=0.0
        cdef double toa_lw_up_clear_local =0.0, toa_lw_down_clear_local=0.0, toa_sw_up_clear_local=0.0, toa_sw_down_clear_local=0.0

        cdef double nxny_i = 1.0/(Gr.dims.n[0]*Gr.dims.n[1])
        with nogil:
           for ip in xrange(n_pencils):
               srf_lw_up_local   += uflx_lw_out[ip,0] * nxny_i
               srf_lw_down_local += dflx_lw_out[ip,0] * nxny_i
               srf_sw_up_local   +=  uflx_sw_out[ip,0] * nxny_i
               srf_sw_down_local += dflx_sw_out[ip,0] * nxny_i

               toa_lw_up_local   += uflx_lw_out[ip,nz_full] * nxny_i
               toa_lw_down_local += dflx_lw_out[ip,nz_full] * nxny_i
               toa_sw_up_local   +=  uflx_sw_out[ip,nz_full] * nxny_i
               toa_sw_down_local += dflx_sw_out[ip,nz_full] * nxny_i

               srf_lw_up_clear_local   += uflxc_lw_out[ip,0] * nxny_i
               srf_lw_down_clear_local += dflxc_lw_out[ip,0] * nxny_i
               srf_sw_up_clear_local   +=  uflxc_sw_out[ip,0] * nxny_i
               srf_sw_down_clear_local += dflxc_sw_out[ip,0] * nxny_i

               toa_lw_up_clear_local   += uflxc_lw_out[ip,nz_full] * nxny_i
               toa_lw_down_clear_local += dflxc_lw_out[ip,nz_full] * nxny_i
               toa_sw_up_clear_local   +=  uflxc_sw_out[ip,nz_full] * nxny_i
               toa_sw_down_clear_local += dflxc_sw_out[ip,nz_full] * nxny_i

               for k in xrange(nz):
                   heating_rate_pencil[ip, k] = (hr_lw_out[ip,k] + hr_sw_out[ip,k]) * Ref.rho0_half_global[k+gw] * cpm_c(qv_pencil[ip,k])/86400.0
                   uflux_lw_pencil[ip,k] = (uflx_lw_out[ip,k] + uflx_lw_out[ip, k+1]) * 0.5
                   dflux_lw_pencil[ip,k] = (dflx_lw_out[ip,k] + dflx_lw_out[ip, k+1]) * 0.5
                   uflux_sw_pencil[ip,k] = (uflx_sw_out[ip,k] + uflx_sw_out[ip, k+1]) * 0.5
                   dflux_sw_pencil[ip,k] = (dflx_sw_out[ip,k] + dflx_sw_out[ip, k+1]) * 0.5

                   heating_rate_clear_pencil[ip, k] = (hrc_lw_out[ip,k] + hrc_sw_out[ip,k]) * Ref.rho0_half_global[k+gw] * cpm_c(qv_pencil[ip,k])/86400.0
                   uflux_lw_clear_pencil[ip,k] = (uflxc_lw_out[ip,k] + uflxc_lw_out[ip, k+1]) * 0.5
                   dflux_lw_clear_pencil[ip,k] = (dflxc_lw_out[ip,k] + dflxc_lw_out[ip, k+1]) * 0.5
                   uflux_sw_clear_pencil[ip,k] = (uflxc_sw_out[ip,k] + uflxc_sw_out[ip, k+1]) * 0.5
                   dflux_sw_clear_pencil[ip,k] = (dflxc_sw_out[ip,k] + dflxc_sw_out[ip, k+1]) * 0.5

        self.srf_lw_up = Pa.domain_scalar_sum(srf_lw_up_local)
        self.srf_lw_down = Pa.domain_scalar_sum(srf_lw_down_local)
        self.srf_sw_up= Pa.domain_scalar_sum(srf_sw_up_local)
        self.srf_sw_down= Pa.domain_scalar_sum(srf_sw_down_local)

        self.toa_lw_up = Pa.domain_scalar_sum(toa_lw_up_local)
        self.toa_lw_down = Pa.domain_scalar_sum(toa_lw_down_local)
        self.toa_sw_up= Pa.domain_scalar_sum(toa_sw_up_local)
        self.toa_sw_down= Pa.domain_scalar_sum(toa_sw_down_local)

        self.srf_lw_up_clear = Pa.domain_scalar_sum(srf_lw_up_clear_local)
        self.srf_lw_down_clear = Pa.domain_scalar_sum(srf_lw_down_clear_local)
        self.srf_sw_up_clear = Pa.domain_scalar_sum(srf_sw_up_clear_local)
        self.srf_sw_down_clear = Pa.domain_scalar_sum(srf_sw_down_clear_local)

        self.toa_lw_up_clear = Pa.domain_scalar_sum(toa_lw_up_clear_local)
        self.toa_lw_down_clear = Pa.domain_scalar_sum(toa_lw_down_clear_local)
        self.toa_sw_up_clear = Pa.domain_scalar_sum(toa_sw_up_clear_local)
        self.toa_sw_down_clear = Pa.domain_scalar_sum(toa_sw_down_clear_local)

        self.z_pencil.reverse_double(&Gr.dims, Pa, heating_rate_pencil, &self.heating_rate[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, uflux_lw_pencil, &self.uflux_lw[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, dflux_lw_pencil, &self.dflux_lw[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, uflux_sw_pencil, &self.uflux_sw[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, dflux_sw_pencil, &self.dflux_sw[0])

        self.z_pencil.reverse_double(&Gr.dims, Pa, heating_rate_clear_pencil, &self.heating_rate_clear[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, uflux_lw_clear_pencil, &self.uflux_lw_clear[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, dflux_lw_clear_pencil, &self.dflux_lw_clear[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, uflux_sw_clear_pencil, &self.uflux_sw_clear[0])
        self.z_pencil.reverse_double(&Gr.dims, Pa, dflux_sw_clear_pencil, &self.dflux_sw_clear[0])
        return


    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        RadiationBase.stats_io(self, Gr, Ref, DV, NS, Pa)


        return

#Calculate cos(solar zenith angle) from radiation.f90 in JPLLES provided by Colleen
cdef double cos_sza(double jday, double hourz, double dlat, double dlon) nogil:

    cdef double epsiln = 0.016733
    cdef double sinob = 0.3978
    cdef double dpy = 365.242 #degrees per year
    cdef double dph = 15.0 #degrees per hour
    cdef double day_angle = 2.0*pi*(jday-1.)/dpy
    #Hours of Meridian Passage (true solar noon)
    cdef double homp = (12.0 + 0.12357*sin(day_angle) - 0.004289*cos(day_angle) + 0.153809*sin(2*day_angle) + 0.060783*cos(2*day_angle))
    cdef double hour_angle = dph*(hourz - homp) - dlon
    cdef double ang = 279.9348*pi/180. + day_angle
    cdef double sigma = (ang*180./pi + 0.4087*sin(ang) + 1.8724*cos(ang) - 0.0182*sin(2.*ang) + 0.0083*cos(2.*ang))*pi/180.
    cdef double sindlt = sinob*sin(sigma)
    cdef double cosdlt = sqrt(1. - sindlt*sindlt)
    cdef double cos_sza = fmax(sindlt*sin(pi/180.*dlat) +cosdlt*cos(pi/180.*dlat)*cos(pi/180.*hour_angle), 0.0)

    return cos_sza

def get_humidity(temperature_old, qt_old, pressure, temperature_new, Th):
    pv_star_1 = Th.get_pv_star(temperature_old)
    pv_1 = (pressure * qt_old) / (eps_v * (1.0 - qt_old) + qt_old)
    rh_ = pv_1 / pv_star_1
    pv_star_2 = Th.get_pv_star(temperature_new)
    pv_2 = rh_ * pv_star_2
    qt_new = 1.0/(eps_vi * (pressure - pv_2)/pv_2 + 1.0)
    return qt_new