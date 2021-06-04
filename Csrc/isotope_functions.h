#pragma once
#include "parameters.h"
#include <math.h>

// fractionation factor α_eq for 018 is based equations from Majoube 1971
static inline double equilibrium_fractionation_factor_H2O18(double t){
	double alpha_tmp = exp(1137/(t*t) - 0.4156/t -2.0667e-3);  
    return alpha_tmp;
}

// Rayleigh distillation is adopted from Wei's paper in 2018 for qt_iso initialization
static inline double Rayleigh_distillation(double qt){
    double delta;
    double R;
    delta = 8.99 * log((qt*1000)/0.622) - 42.9;
    R = (delta/1000 + 1) * R_std_O18;
    return R*qt;
}

// calculate delta of specific water phase variable, values of isotopeic varialbe is after scaled.
static inline double q_2_delta(double const q_iso, double const q){
    return ((q_iso/q) - 1) * 1000;
}

static inline void calculate_delta_using_q(double const qt, double const qv, double const ql, 
                                            double const qt_iso, double const qv_iso, double const ql_iso, 
                                            double* delta_qt, double* delta_qv, double* delta_ql){
    * delta_qt = q_2_delta(qt_iso, qt);
    * delta_qv = q_2_delta(qv_iso, qv);
    if(ql > 0.0){
        * delta_ql = q_2_delta(ql_iso, ql);
    }
    return;
}

// No microphysics fractioantion based on Wei's 2018
static inline void iso_eq_frac_NoMicro_function(double const qt_iso, double const qv, double const ql, 
    double* qv_iso, double* ql_iso, double const alpha_eq){
    
    double ql_iso_tmp;
    double qv_iso_tmp;
    if (ql > 0.0){
        ql_iso_tmp = qt_iso / (1.0+(qv/ql)*(1.0/alpha_eq));
        qv_iso_tmp = qt_iso - ql_iso_tmp;
    }
    else{
        ql_iso_tmp = 0.0;
        qv_iso_tmp = qt_iso;
    }
    * ql_iso = ql_iso_tmp;
    * qv_iso = qv_iso_tmp;
    return;
}
