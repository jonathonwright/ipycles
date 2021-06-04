#pragma once
#include "parameters.h"
#include "isotope_functions.h"
#include <math.h>

void iso_equilibrium_fractionation_No_Microphysics(struct DimStruct *dims, 
    int water_type, double* restrict t, double* restrict qt, double* restrict qv, double* restrict ql, 
    double* restrict qt_iso, double* restrict qv_iso, double* restrict ql_iso){
    
    ssize_t i,j,k;
    double alpha_eq = 0.0;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    // water type = 0, water tracer, fractionation factor = 1.0
                    if(water_type == 0){
                        alpha_eq = 1.0;
                    }
                    // water type = 1, HDO, fractionation factor of HDO 
                    else if(water_type == 1){
                        alpha_eq = 1.0;
                    }
                    // water type = 2, H2O18, fractionation factor of H2O18 
                    else if(water_type == 2){
                        alpha_eq = equilibrium_fractionation_factor_H2O18(t[ijk]);
                    }
                    iso_eq_frac_NoMicro_function(qt_iso[ijk], qv[ijk], ql[ijk], &qv_iso[ijk], &ql_iso[ijk], alpha_eq);
                } // End k loop
            } // End j loop
        } // End i loop
    return;
}

void delta_isotopologue(struct DimStruct *dims, double* restrict qt, double* restrict qv, double* restrict ql, 
    double* restrict qt_iso, double* restrict qv_iso, double* restrict ql_iso, 
    double* restrict delta_qt, double* restrict delta_qv, double* restrict delta_ql){
    
    ssize_t i,j,k;
    double alpha_eq = 0.0;
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const ssize_t ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const ssize_t jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const ssize_t ijk = ishift + jshift + k;
                    calculate_delta_using_q(qt[ijk], qv[ijk], ql[ijk], qt_iso[ijk], 
                                            qv_iso[ijk], ql_iso[ijk], 
                                            &delta_qt[ijk], &delta_qv[ijk], &delta_ql[ijk]);
                } // End k loop
            } // End j loop
        } // End i loop
    return;
}

// Scaling the isotope specific humidity values back to correct magnitude
void statsIO_isotope_scaling_magnitude(struct DimStruct *dims, double* restrict tmp_values){
    ssize_t i;
    const ssize_t imin = 0;
    const ssize_t imax = dims->nlg[2];
    for (i=imin; i<imax; i++){
        tmp_values[i] *= R_std_O18;
    } 
    return;
}