"""
CEC2017 Test Function Suite for Single Objective Optimization - Python version
Fully faithful translation of the original C++ code by Noor Awad.
All functions internally call sr_func (shift/rotate) as in the original.
"""

import numpy as np
import os
import math

# Constants
INF = 1.0e99
EPS = 1.0e-14
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029

# Global cache for loaded data (keyed by (func_num, nx))
_data_cache = {}

# ------------------------------------------------------------------------------
# Auxiliary functions (shift, rotate, etc.)
# ------------------------------------------------------------------------------

def shiftfunc(x, Os):
    """Shift by subtracting the global optimum."""
    return x - Os

def rotatefunc(x, Mr, nx):
    """Rotate using matrix Mr (size nx*nx)."""
    # Mr is expected to be a 2D numpy array of shape (nx, nx)
    return np.dot(Mr, x)

def sr_func(x, Os, Mr, sh_rate, s_flag, r_flag, nx):
    """
    Shift and/or rotate with scaling.
    Returns the transformed vector.
    """
    if s_flag == 1:
        y = shiftfunc(x, Os)
        y = y * sh_rate
        if r_flag == 1:
            return rotatefunc(y, Mr, nx)
        else:
            return y
    else:
        if r_flag == 1:
            y = x * sh_rate
            return rotatefunc(y, Mr, nx)
        else:
            return x * sh_rate

def asyfunc(x, beta, nx):
    """Asymmetry transformation."""
    xasy = np.copy(x)
    for i in range(nx):
        if x[i] > 0:
            xasy[i] = pow(x[i], 1.0 + beta * i / (nx - 1) * pow(x[i], 0.5))
    return xasy

def oszfunc(x, nx):
    """Oscillatory transformation."""
    xosz = np.copy(x)
    for i in range(nx):
        if i == 0 or i == nx - 1:
            if x[i] != 0:
                xx = math.log(abs(x[i]))
                if x[i] > 0:
                    c1, c2 = 10.0, 7.9
                else:
                    c1, c2 = 5.5, 3.1
                sx = 1 if x[i] > 0 else (-1 if x[i] < 0 else 0)
                xosz[i] = sx * math.exp(xx + 0.049 * (math.sin(c1 * xx) + math.sin(c2 * xx)))
            else:
                xosz[i] = 0
        else:
            xosz[i] = x[i]
    return xosz

def cf_cal(x, Os, delta, bias, fit, cf_num, nx):
    """
    Composition function calculation.
    fit: array of function values for each component (already added bias)
    """
    w = np.zeros(cf_num)
    for i in range(cf_num):
        fit[i] += bias[i]
        d = np.sum((x - Os[i*nx:(i+1)*nx])**2)
        if d != 0:
            w[i] = (np.sqrt(1.0 / d)) * np.exp(-d / (2.0 * nx * (delta[i]**2)))
        else:
            w[i] = INF
    w_max = np.max(w)
    w_sum = np.sum(w)
    if w_max == 0:
        w[:] = 1.0
        w_sum = cf_num
    f = np.sum(w / w_sum * fit)
    return f

# ------------------------------------------------------------------------------
# Basic functions (each receives raw x and transformation data, performs sr_func)
# ------------------------------------------------------------------------------

def sphere_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    return np.sum(z**2)

def ellips_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    w = np.power(10.0, 6.0 * np.arange(nx) / (nx - 1))
    return np.sum(w * z**2)

def sum_diff_pow_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    s = 0.0
    for i in range(nx):
        s += np.abs(z[i]) ** (i + 1)
    return s

def zakharov_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    sum1 = np.sum(z**2)
    sum2 = np.sum(0.5 * (np.arange(1, nx+1)) * z)
    return sum1 + sum2**2 + sum2**4

def levy_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    w = 1.0 + (z - 1.0) / 4.0
    term1 = np.sin(PI * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * PI * w[-1])**2)
    s = 0.0
    for i in range(nx-1):
        s += (w[i] - 1)**2 * (1 + 10 * np.sin(PI * w[i] + 1)**2)
    return term1 + s + term3

def dixon_price_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    term1 = (z[0] - 1)**2
    s = 0.0
    for i in range(1, nx):
        s += i * (2 * z[i]**2 - z[i-1])**2
    return term1 + s

def bent_cigar_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    return z[0]**2 + 1e6 * np.sum(z[1:]**2)

def discus_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    return 1e6 * z[0]**2 + np.sum(z[1:]**2)

def dif_powers_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    p = 2 + 4 * np.arange(nx) / (nx - 1)
    return np.sqrt(np.sum(np.abs(z) ** p))

def rosenbrock_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 2.048/100.0, s_flag, r_flag, nx)
    z = z + 1.0  # shift to origin
    s = 0.0
    for i in range(nx-1):
        s += 100.0 * (z[i]**2 - z[i+1])**2 + (z[i] - 1.0)**2
    return s

def schaffer_F7_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    s = 0.0
    for i in range(nx-1):
        zi = np.sqrt(z[i]*z[i] + z[i+1]*z[i+1])
        tmp = np.sin(50.0 * zi**0.2)
        s += np.sqrt(zi) + np.sqrt(zi) * (tmp**2)
    return (s / (nx-1))**2

def ackley_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    sum1 = np.sum(z**2)
    sum2 = np.sum(np.cos(2 * PI * z))
    return E - 20.0 * np.exp(-0.2 * np.sqrt(sum1 / nx)) - np.exp(sum2 / nx) + 20.0

def rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 5.12/100.0, s_flag, r_flag, nx)
    return np.sum(z**2 - 10.0 * np.cos(2 * PI * z) + 10.0)

def step_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    # First, shift and scale without rotation
    if s_flag == 1:
        y = shiftfunc(x, Os)
    else:
        y = np.copy(x)
    y = y * 5.12/100.0
    # Apply non-continuous transformation
    for i in range(nx):
        if np.abs(y[i] - Os[i]) > 0.5:   # Note: in original, they compare y[i] with Os[i] after shift? Actually original code uses 'y' which is the shifted vector. We need Os for the rounding reference.
            # But in original, they have a separate step: if fabs(y[i]-Os[i])>0.5, set y[i] = Os[i] + floor(2*(y[i]-Os[i])+0.5)/2
            # Here y is already shifted? Let's follow original exactly.
            # We have y = x - Os (if s_flag). So y[i] - Os[i] = x[i] - 2*Os[i]? That seems odd.
            # Actually the original code:
            # if (fabs(y[i]-Os[i])>0.5)
            #   y[i]=Os[i]+floor(2*(y[i]-Os[i])+0.5)/2;
            # Here y is from shiftfunc (x-Os). So y[i]-Os[i] = x[i] - 2*Os[i].
            # This is likely a bug, but we must replicate exactly.
            # So we'll use the original formula.
            diff = y[i] - Os[i]
            if np.abs(diff) > 0.5:
                y[i] = Os[i] + np.floor(2 * diff + 0.5) / 2.0
    # Now apply rotation if required
    if r_flag == 1:
        z = rotatefunc(y, Mr, nx)
    else:
        z = y
    # Compute Rastrigin
    return np.sum(z**2 - 10.0 * np.cos(2 * PI * z) + 10.0)

def schwefel_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1000.0/100.0, s_flag, r_flag, nx)
    z = z + 4.209687462275036e+002
    f = 0.0
    for i in range(nx):
        if z[i] > 500:
            f -= (500.0 - np.fmod(z[i], 500)) * np.sin(np.sqrt(500.0 - np.fmod(z[i], 500)))
            tmp = (z[i] - 500.0) / 100.0
            f += tmp * tmp / nx
        elif z[i] < -500:
            f -= (-500.0 + np.fmod(np.abs(z[i]), 500)) * np.sin(np.sqrt(500.0 - np.fmod(np.abs(z[i]), 500)))
            tmp = (z[i] + 500.0) / 100.0
            f += tmp * tmp / nx
        else:
            f -= z[i] * np.sin(np.sqrt(np.abs(z[i])))
    return f + 4.189828872724338e+002 * nx

def katsuura_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    tmp3 = nx ** 1.2
    prod = 1.0
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = 2.0 ** j
            tmp2 = tmp1 * z[i]
            temp += np.abs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
        prod *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
    tmp1 = 10.0 / nx / nx
    return prod * tmp1 - tmp1

def bi_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    # Step 1: shift and scale (but not rotate yet)
    if s_flag == 1:
        y = shiftfunc(x, Os)
    else:
        y = np.copy(x)
    y = y * 10.0/100.0   # shrink

    # tmpx = 2*y, then adjust sign based on Os
    tmpx = 2.0 * y
    for i in range(nx):
        if Os[i] < 0.0:
            tmpx[i] *= -1.0

    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * np.sqrt(nx + 20.0) - 8.2)
    mu1 = -np.sqrt((mu0*mu0 - d) / s)

    # Compute tmp1 and tmp2 using tmpx
    tmp1 = np.sum((tmpx - mu0)**2)
    tmp2 = np.sum((tmpx - mu1)**2) * s + d * nx

    # Now rotation for the cos part
    if r_flag == 1:
        # rotate tmpx to get z
        z = rotatefunc(tmpx, Mr, nx)
    else:
        z = tmpx

    cos_sum = np.sum(np.cos(2 * PI * z))
    return min(tmp1, tmp2) + 10.0 * (nx - cos_sum)

def grie_rosen_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    z = z + 1.0   # shift to origin
    f = 0.0
    for i in range(nx-1):
        tmp1 = z[i]**2 - z[i+1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1**2 + tmp2**2
        f += (temp**2)/4000.0 - np.cos(temp) + 1.0
    tmp1 = z[-1]**2 - z[0]
    tmp2 = z[-1] - 1.0
    temp = 100.0 * tmp1**2 + tmp2**2
    f += (temp**2)/4000.0 - np.cos(temp) + 1.0
    return f

def escaffer6_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = 0.0
    for i in range(nx-1):
        temp1 = np.sin(np.sqrt(z[i]**2 + z[i+1]**2))**2
        temp2 = 1.0 + 0.001 * (z[i]**2 + z[i+1]**2)
        f += 0.5 + (temp1 - 0.5) / temp2**2
    temp1 = np.sin(np.sqrt(z[-1]**2 + z[0]**2))**2
    temp2 = 1.0 + 0.001 * (z[-1]**2 + z[0]**2)
    f += 0.5 + (temp1 - 0.5) / temp2**2
    return f

def happycat_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    z = z - 1.0   # shift to origin
    r2 = np.sum(z**2)
    sum_z = np.sum(z)
    alpha = 1.0/8.0
    return np.abs(r2 - nx)**(2*alpha) + (0.5 * r2 + sum_z) / nx + 0.5

def hgbat_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    z = z - 1.0
    r2 = np.sum(z**2)
    sum_z = np.sum(z)
    alpha = 1.0/4.0
    return np.abs(r2**2 - sum_z**2)**(2*alpha) + (0.5 * r2 + sum_z) / nx + 0.5

# ------------------------------------------------------------------------------
# Hybrid functions
# ------------------------------------------------------------------------------

def hf01(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 3
    Gp = [0.2, 0.4, 0.4]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    # shift and rotate (global)
    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)

    # permute dimensions using SS (1-indexed)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = zakharov_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = rosenbrock_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = rastrigin_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    return np.sum(fit)

def hf02(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 3
    Gp = [0.3, 0.3, 0.4]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = ellips_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = schwefel_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = bent_cigar_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    return np.sum(fit)

def hf03(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 3
    Gp = [0.3, 0.3, 0.4]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = bent_cigar_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = rosenbrock_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = bi_rastrigin_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    return np.sum(fit)

def hf04(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 4
    Gp = [0.2, 0.2, 0.2, 0.4]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = ellips_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = ackley_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = schaffer_F7_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    fit[3] = rastrigin_func(y[G[3]:G[3]+G_nx[3]], G_nx[3], Os, Mr, 0, 0)
    return np.sum(fit)

def hf05(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 4
    Gp = [0.2, 0.2, 0.3, 0.3]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = bent_cigar_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = hgbat_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = rastrigin_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    fit[3] = rosenbrock_func(y[G[3]:G[3]+G_nx[3]], G_nx[3], Os, Mr, 0, 0)
    return np.sum(fit)

def hf06(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 4
    Gp = [0.2, 0.2, 0.3, 0.3]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = escaffer6_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = hgbat_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = rosenbrock_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    fit[3] = schwefel_func(y[G[3]:G[3]+G_nx[3]], G_nx[3], Os, Mr, 0, 0)
    return np.sum(fit)

def hf07(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 5
    Gp = [0.1, 0.2, 0.2, 0.2, 0.3]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = katsuura_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = ackley_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = grie_rosen_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    fit[3] = schwefel_func(y[G[3]:G[3]+G_nx[3]], G_nx[3], Os, Mr, 0, 0)
    fit[4] = rastrigin_func(y[G[4]:G[4]+G_nx[4]], G_nx[4], Os, Mr, 0, 0)
    return np.sum(fit)

def hf08(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 5
    Gp = [0.2, 0.2, 0.2, 0.2, 0.2]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = ellips_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = ackley_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = rastrigin_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    fit[3] = hgbat_func(y[G[3]:G[3]+G_nx[3]], G_nx[3], Os, Mr, 0, 0)
    fit[4] = discus_func(y[G[4]:G[4]+G_nx[4]], G_nx[4], Os, Mr, 0, 0)
    return np.sum(fit)

def hf09(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 5
    Gp = [0.2, 0.2, 0.2, 0.2, 0.2]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = bent_cigar_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = rastrigin_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = grie_rosen_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    fit[3] = weierstrass_func(y[G[3]:G[3]+G_nx[3]], G_nx[3], Os, Mr, 0, 0)
    fit[4] = escaffer6_func(y[G[4]:G[4]+G_nx[4]], G_nx[4], Os, Mr, 0, 0)
    return np.sum(fit)

def hf10(x, nx, Os, Mr, SS, s_flag, r_flag):
    cf_num = 6
    Gp = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
    G_nx = [0]*cf_num
    tmp = 0
    for i in range(cf_num-1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num-1] = nx - tmp
    G = [0]*cf_num
    for i in range(1, cf_num):
        G[i] = G[i-1] + G_nx[i-1]

    z = sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[SS[i]-1]

    fit = np.zeros(cf_num)
    fit[0] = hgbat_func(y[G[0]:G[0]+G_nx[0]], G_nx[0], Os, Mr, 0, 0)
    fit[1] = katsuura_func(y[G[1]:G[1]+G_nx[1]], G_nx[1], Os, Mr, 0, 0)
    fit[2] = ackley_func(y[G[2]:G[2]+G_nx[2]], G_nx[2], Os, Mr, 0, 0)
    fit[3] = rastrigin_func(y[G[3]:G[3]+G_nx[3]], G_nx[3], Os, Mr, 0, 0)
    fit[4] = schwefel_func(y[G[4]:G[4]+G_nx[4]], G_nx[4], Os, Mr, 0, 0)
    fit[5] = schaffer_F7_func(y[G[0]:G[0]+G_nx[5]], G_nx[5], Os, Mr, 0, 0)
    return np.sum(fit)

# ------------------------------------------------------------------------------
# Composition functions
# ------------------------------------------------------------------------------

def cf01(x, nx, Os, Mr, r_flag):
    cf_num = 3
    delta = np.array([10, 20, 30])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)

    # Note: Os and Mr are stacked for each component
    # Os shape: (cf_num, nx)
    # Mr shape: (cf_num, nx, nx)
    fit[0] = rosenbrock_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[1] = ellips_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[1] = 10000 * fit[1] / 1e+10
    fit[2] = rastrigin_func(x, nx, Os[2], Mr[2], 1, r_flag)
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf02(x, nx, Os, Mr, r_flag):
    cf_num = 3
    delta = np.array([10, 20, 30])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)
    fit[0] = rastrigin_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[1] = griewank_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[1] = 1000 * fit[1] / 100
    fit[2] = schwefel_func(x, nx, Os[2], Mr[2], 1, r_flag)
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf03(x, nx, Os, Mr, r_flag):
    cf_num = 4
    delta = np.array([10, 20, 30, 40])
    bias = np.array([0, 100, 200, 300])
    fit = np.zeros(cf_num)
    fit[0] = rosenbrock_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[1] = ackley_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[1] = 1000 * fit[1] / 100
    fit[2] = schwefel_func(x, nx, Os[2], Mr[2], 1, r_flag)
    fit[3] = rastrigin_func(x, nx, Os[3], Mr[3], 1, r_flag)
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf04(x, nx, Os, Mr, r_flag):
    cf_num = 4
    delta = np.array([10, 20, 30, 40])
    bias = np.array([0, 100, 200, 300])
    fit = np.zeros(cf_num)
    fit[0] = ackley_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[0] = 1000 * fit[0] / 100
    fit[1] = ellips_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[1] = 10000 * fit[1] / 1e+10
    fit[2] = griewank_func(x, nx, Os[2], Mr[2], 1, r_flag)
    fit[2] = 1000 * fit[2] / 100
    fit[3] = rastrigin_func(x, nx, Os[3], Mr[3], 1, r_flag)
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf05(x, nx, Os, Mr, r_flag):
    cf_num = 5
    delta = np.array([10, 20, 30, 40, 50])
    bias = np.array([0, 100, 200, 300, 400])
    fit = np.zeros(cf_num)
    fit[0] = rastrigin_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[0] = 10000 * fit[0] / 1e+3
    fit[1] = happycat_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[1] = 1000 * fit[1] / 1e+3
    fit[2] = ackley_func(x, nx, Os[2], Mr[2], 1, r_flag)
    fit[2] = 1000 * fit[2] / 100
    fit[3] = discus_func(x, nx, Os[3], Mr[3], 1, r_flag)
    fit[3] = 10000 * fit[3] / 1e+10
    fit[4] = rosenbrock_func(x, nx, Os[4], Mr[4], 1, r_flag)
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf06(x, nx, Os, Mr, r_flag):
    cf_num = 5
    delta = np.array([10, 20, 20, 30, 40])
    bias = np.array([0, 100, 200, 300, 400])
    fit = np.zeros(cf_num)
    fit[0] = escaffer6_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[0] = 10000 * fit[0] / 2e+7
    fit[1] = schwefel_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[2] = griewank_func(x, nx, Os[2], Mr[2], 1, r_flag)
    fit[2] = 1000 * fit[2] / 100
    fit[3] = rosenbrock_func(x, nx, Os[3], Mr[3], 1, r_flag)
    fit[4] = rastrigin_func(x, nx, Os[4], Mr[4], 1, r_flag)
    fit[4] = 10000 * fit[4] / 1e+3
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf07(x, nx, Os, Mr, r_flag):
    cf_num = 6
    delta = np.array([10, 20, 30, 40, 50, 60])
    bias = np.array([0, 100, 200, 300, 400, 500])
    fit = np.zeros(cf_num)
    fit[0] = hgbat_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[0] = 10000 * fit[0] / 1000
    fit[1] = rastrigin_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[1] = 10000 * fit[1] / 1e+3
    fit[2] = schwefel_func(x, nx, Os[2], Mr[2], 1, r_flag)
    fit[2] = 10000 * fit[2] / 4e+3
    fit[3] = bent_cigar_func(x, nx, Os[3], Mr[3], 1, r_flag)
    fit[3] = 10000 * fit[3] / 1e+30
    fit[4] = ellips_func(x, nx, Os[4], Mr[4], 1, r_flag)
    fit[4] = 10000 * fit[4] / 1e+10
    fit[5] = escaffer6_func(x, nx, Os[5], Mr[5], 1, r_flag)
    fit[5] = 10000 * fit[5] / 2e+7
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf08(x, nx, Os, Mr, r_flag):
    cf_num = 6
    delta = np.array([10, 20, 30, 40, 50, 60])
    bias = np.array([0, 100, 200, 300, 400, 500])
    fit = np.zeros(cf_num)
    fit[0] = ackley_func(x, nx, Os[0], Mr[0], 1, r_flag)
    fit[0] = 1000 * fit[0] / 100
    fit[1] = griewank_func(x, nx, Os[1], Mr[1], 1, r_flag)
    fit[1] = 1000 * fit[1] / 100
    fit[2] = discus_func(x, nx, Os[2], Mr[2], 1, r_flag)
    fit[2] = 10000 * fit[2] / 1e+10
    fit[3] = rosenbrock_func(x, nx, Os[3], Mr[3], 1, r_flag)
    fit[4] = happycat_func(x, nx, Os[4], Mr[4], 1, r_flag)
    fit[4] = 1000 * fit[4] / 1e+3
    fit[5] = escaffer6_func(x, nx, Os[5], Mr[5], 1, r_flag)
    fit[5] = 10000 * fit[5] / 2e+7
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf09(x, nx, Os, Mr, SS, r_flag):
    cf_num = 3
    delta = np.array([10, 30, 50])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)
    fit[0] = hf05(x, nx, Os[0], Mr[0], SS[0], 1, r_flag)
    fit[1] = hf06(x, nx, Os[1], Mr[1], SS[1], 1, r_flag)
    fit[2] = hf07(x, nx, Os[2], Mr[2], SS[2], 1, r_flag)
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

def cf10(x, nx, Os, Mr, SS, r_flag):
    cf_num = 3
    delta = np.array([10, 30, 50])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)
    fit[0] = hf05(x, nx, Os[0], Mr[0], SS[0], 1, r_flag)
    fit[1] = hf08(x, nx, Os[1], Mr[1], SS[1], 1, r_flag)
    fit[2] = hf09(x, nx, Os[2], Mr[2], SS[2], 1, r_flag)
    return cf_cal(x, Os.flatten(), delta, bias, fit, cf_num, nx)

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

def cec17_test_func(x, func_num):
    """
    Evaluate CEC2017 test function.

    Parameters
    ----------
    x : np.ndarray, shape (nx, mx)
        Input matrix where each column is a point (nx dimensions, mx points).
    func_num : int
        Function number (1 to 30).

    Returns
    -------
    f : np.ndarray, shape (mx,)
        Function values for each point.
    """
    nx, mx = x.shape
    # Check dimension validity
    if nx not in (2, 10, 20, 30, 50, 100):
        raise ValueError("Test functions are only defined for D=2,10,20,30,50,100.")
    if nx == 2 and (17 <= func_num <= 22 or 29 <= func_num <= 30):
        raise ValueError(f"Function F{func_num} is not defined for D=2.")

    # Load data if not cached
    key = (func_num, nx)
    if key not in _data_cache:
        load_data(func_num, nx)
    data = _data_cache[key]
    OShift = data['OShift']
    M = data['M']
    SS = data.get('SS', None)

    f = np.zeros(mx)
    for i in range(mx):
        xi = x[:, i]
        if func_num == 1:
            val = bent_cigar_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 100.0
        elif func_num == 2:
            raise ValueError("Function F2 has been deleted.")
        elif func_num == 3:
            val = zakharov_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 300.0
        elif func_num == 4:
            val = rosenbrock_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 400.0
        elif func_num == 5:
            val = rastrigin_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 500.0
        elif func_num == 6:
            val = schaffer_F7_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 600.0
        elif func_num == 7:
            val = bi_rastrigin_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 700.0
        elif func_num == 8:
            val = step_rastrigin_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 800.0
        elif func_num == 9:
            val = levy_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 900.0
        elif func_num == 10:
            val = schwefel_func(xi, nx, OShift, M, 1, 1)
            f[i] = val + 1000.0
        elif func_num == 11:
            val = hf01(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1100.0
        elif func_num == 12:
            val = hf02(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1200.0
        elif func_num == 13:
            val = hf03(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1300.0
        elif func_num == 14:
            val = hf04(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1400.0
        elif func_num == 15:
            val = hf05(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1500.0
        elif func_num == 16:
            val = hf06(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1600.0
        elif func_num == 17:
            val = hf07(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1700.0
        elif func_num == 18:
            val = hf08(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1800.0
        elif func_num == 19:
            val = hf09(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 1900.0
        elif func_num == 20:
            val = hf10(xi, nx, OShift, M, SS, 1, 1)
            f[i] = val + 2000.0
        elif func_num == 21:
            val = cf01(xi, nx, OShift, M, 1)
            f[i] = val + 2100.0
        elif func_num == 22:
            val = cf02(xi, nx, OShift, M, 1)
            f[i] = val + 2200.0
        elif func_num == 23:
            val = cf03(xi, nx, OShift, M, 1)
            f[i] = val + 2300.0
        elif func_num == 24:
            val = cf04(xi, nx, OShift, M, 1)
            f[i] = val + 2400.0
        elif func_num == 25:
            val = cf05(xi, nx, OShift, M, 1)
            f[i] = val + 2500.0
        elif func_num == 26:
            val = cf06(xi, nx, OShift, M, 1)
            f[i] = val + 2600.0
        elif func_num == 27:
            val = cf07(xi, nx, OShift, M, 1)
            f[i] = val + 2700.0
        elif func_num == 28:
            val = cf08(xi, nx, OShift, M, 1)
            f[i] = val + 2800.0
        elif func_num == 29:
            val = cf09(xi, nx, OShift, M, SS, 1)
            f[i] = val + 2900.0
        elif func_num == 30:
            val = cf10(xi, nx, OShift, M, SS, 1)
            f[i] = val + 3000.0
        else:
            raise ValueError("Function number must be between 1 and 30.")
    if func_num > 2:
        return f - 100
    else:
        return f

def load_data(func_num, nx):
    """Load data files for given function and dimension."""
    cf_num = 10  # for composition functions with multiple matrices
    base_path = "input_data"
    key = (func_num, nx)
    data = {}

    # Load rotation matrix M
    if func_num <= 20:
        fname = os.path.join(base_path, f"M_{func_num}_D{nx}.txt")
        M = np.loadtxt(fname).reshape(nx, nx)
        data['M'] = M
    else:
        fname = os.path.join(base_path, f"M_{func_num}_D{nx}.txt")
        M_all = np.loadtxt(fname).flatten()[:cf_num * nx * nx]
        # It should have cf_num*nx*nx elements
        M = M_all.reshape(cf_num, nx, nx)
        data['M'] = M

    # Load shift data
    fname = os.path.join(base_path, f"shift_data_{func_num}.txt")
    if func_num <= 20:
        OShift = np.loadtxt(fname)
        # Ensure it's a 1D array of length nx
        if OShift.size > nx:
            OShift = OShift[:nx]
        data['OShift'] = OShift
    else:
        OShift_all = np.loadtxt(fname)[:, :nx]
        # Expected shape (cf_num, nx)
        OShift = OShift_all.reshape(cf_num, nx)
        data['OShift'] = OShift

    # Load shuffle data for hybrid functions
    if 11 <= func_num <= 20:
        fname = os.path.join(base_path, f"shuffle_data_{func_num}_D{nx}.txt")
        SS = np.loadtxt(fname, dtype=int)
        data['SS'] = SS
    elif func_num == 29 or func_num == 30:
        fname = os.path.join(base_path, f"shuffle_data_{func_num}_D{nx}.txt")
        SS = np.loadtxt(fname, dtype=int).reshape(cf_num, nx)
        data['SS'] = SS

    _data_cache[key] = data

# Note: The functions griewank_func and weierstrass_func were omitted in the basic list, add them here.
def griewank_func(x, nx, Os, Mr, s_flag, r_flag):
    z = sr_func(x, Os, Mr, 600.0/100.0, s_flag, r_flag, nx)
    s = np.sum(z**2)
    p = np.prod(np.cos(z / np.sqrt(np.arange(1, nx+1))))
    return 1.0 + s/4000.0 - p

def weierstrass_func(x, nx, Os, Mr, s_flag, r_flag):
    a = 0.5
    b = 3.0
    k_max = 20
    z = sr_func(x, Os, Mr, 0.5/100.0, s_flag, r_flag, nx)
    f = 0.0
    sum2 = 0.0
    for j in range(k_max+1):
        sum2 += a**j * np.cos(2 * PI * b**j * 0.5)
    for i in range(nx):
        s = 0.0
        for j in range(k_max+1):
            s += a**j * np.cos(2 * PI * b**j * (z[i] + 0.5))
        f += s
    return f - nx * sum2

class Task:
    def __init__(self, dim, func_num, lb=-100, hb=100):
        self.D = dim
        self.func_num = func_num
        self.lb = lb
        self.hb = hb

    def function(self, x):
        temp = x * (self.hb - self.lb) + self.lb
        temp = temp[:self.D].reshape(self.D, 1)
        return cec17_test_func(temp, self.func_num)[0]

def Benchmark(i):
    func_nums = [
        19, 13, 7, 9, 18, 22, 6, 1, 10, 4, 17, 26, 11, 23, 30, 14, 5, 25, 3, 15
    ]
    Dims = [
        10, 100, 10, 10, 10, 10, 50, 30, 10, 30, 50, 100, 100, 100, 10, 50, 100, 100, 10, 30
    ]
    return [Task(Dims[2 * i - 2], func_nums[2 * i - 2]),
            Task(Dims[2 * i - 1], func_nums[i * 2 - 1])]