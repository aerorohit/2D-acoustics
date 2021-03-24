#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, vstack, csr_matrix, hstack

def in_U_s(m, n, dx, x0, y0, Mx):
    """" Function to return initial condition for the given problem"""

    x = np.arange(n)*dx+x0
    y = np.arange(m)*dx+y0

    x, y = np.meshgrid(x,y)
    ################################
    # constants for initialization #
    ################################
    eps_a = 0.01
    eps_e = 0.005
    eps_v = 0.007
    alp_a = np.log10(2)/9.0
    alp_e = np.log10(2)/16.0
    alp_v = np.log10(2)/25.0
    x_s_a = 0
    y_s_a = 10
    x_s_e = 67
    y_s_e = 5
    x_s_v = 67
    y_s_v = -6

    #################
    # Tam's problem #
    #################
    ############################
    # eps_a = 0.01             #
    # eps_e = 0.004            #
    # eps_v = 0.0001           #
    # alp_a = np.log10(2)/9.0  #
    # alp_e = np.log10(2)/25   #
    # alp_v = np.log10(2)/25.0 #
    # x_s_a = 0                #
    # y_s_a = 0                #
    # x_s_e = 67               #
    # y_s_e = 0                #
    # x_s_v = 67               #
    # y_s_v = 0                #
    ############################


    # evaluations of initial conditions

    ra2 = (x- x_s_a)**2 + (y-y_s_a)**2
    re2 = (x- x_s_e)**2 + (y-y_s_e)**2
    rv2 = (x- x_s_v)**2 + (y-y_s_v)**2

    t1 = eps_a*np.exp(-alp_a * ra2 )
    t2 = eps_e*np.exp(-alp_e * re2 )
    rho_s = t1+t2

    u_s =  eps_v*(y-y_s_v) * np.exp(-alp_v*rv2)
    v_s = -eps_v*(x-x_s_v) * np.exp(-alp_v*rv2)
    p_s =  eps_a           * np.exp(-alp_a*ra2)

    U_s = np.array([rho_s, u_s, v_s, p_s ])

    return U_s, x, y


def update_E_from_U(U_mat, E_mat, Mx):
    rho_s = U_mat[0]
    u_s =U_mat[1]
    v_s = U_mat[2]
    p_s = U_mat[3]
    E_mat[0] = Mx*rho_s + u_s
    E_mat[1] = Mx*u_s + p_s
    E_mat[2] = Mx*v_s
    E_mat[3] = Mx*p_s + u_s
    return 0

def update_F_from_U(U_mat, F_mat):
    v_s = U_mat[2]
    p_s = U_mat[3]
    F_mat[0] = v_s
    F_mat[2] = p_s
    F_mat[3] = v_s
    return 0


def get_Vtheta(x, y, Mx):
    """ Calculating V(theta),  sin(theta), cos(theta), r for all the gridpoints beforehand"""
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    s = np.sin(theta)
    c = np.cos(theta)
    V = Mx*c + (1- Mx*Mx*s**2 )**0.5
    return V, s, c, r


def get_D(n, stencil, stencil_boundary, len_stencil):
    """ Return: D opetrator
        D.dot(A) gives derivative along columns (y coordinate for this case)
        of A for all points of A

        To get the derivative along rows(x) of the matrix post multiply by
        transpose of D i.e. D.T
    """

    lh = (len_stencil-1)//2

    s_b = csr_matrix(stencil_boundary)
    D = diags(stencil, np.arange(len_stencil), shape = (n-2*lh,n)).toarray()
    zz = csr_matrix((lh, n-len_stencil), dtype=np.int8).toarray()
    s_left = hstack([s_b, zz])
    s_right = hstack([zz, -s_b[::-1,::-1]])
    D = vstack([s_left,D,s_right]).toarray()
    return D


def get_K(U_mat, D, ghost_len, V_mat, s_mat, c_mat, r_mat, Mx):
    """ returns K for interior and boundary points
        where dU/dt = K
    """

    dxU = np.matmul(U_mat, D.T)#U_mat.dot(D.T)
    dyU = np.matmul(D, U_mat)


    dxE = np.zeros_like(U_mat)
    dyF = np.zeros_like(U_mat)
    # we can calculate dxE from dxU in the same way as
    # we calculate E from U as all the operations are linear
    update_E_from_U(dxU, dxE, Mx)
    update_F_from_U(dyU, dyF)

    K = np.zeros_like(U_mat)

    ##############
    # outflow BC #
    ##############
    dt_p_out = V_mat[:, -ghost_len:] * \
        ( -c_mat[:, -ghost_len:]*dxU[3, :, -ghost_len:]
          -s_mat[:, -ghost_len:]*dyU[3, :, -ghost_len:]
          -U_mat[3, :, -ghost_len:]/(2*r_mat[:, -ghost_len:]))


    dt_rho_out = -Mx*dxU[0, :, -ghost_len:] +  (dt_p_out + Mx*dxU[3, :, -ghost_len:])
    dt_u_out = -Mx*dxU[1, :, -ghost_len:] -  dxU[3, :, -ghost_len:]
    dt_v_out = -Mx*dxU[2, :, -ghost_len:] -  dyU[3, :, -ghost_len:]


    K[0, :, -ghost_len:] = dt_rho_out
    K[1, :, -ghost_len:] = dt_u_out
    K[2, :, -ghost_len:] = dt_v_out
    K[3, :, -ghost_len:] = dt_p_out

    ################
    # radiation BC #
    ################
    # inlet
    K[:, :, 0:ghost_len] = V_mat[:, 0:ghost_len] * \
                                ( -c_mat[:, 0:ghost_len]*dxU[:, :, 0:ghost_len]
                                  -s_mat[:, 0:ghost_len]*dyU[:, :, 0:ghost_len]
                                  -U_mat[:, :, 0:ghost_len]/(2*r_mat[:, 0:ghost_len]))
    # top (y here increases as we move down in a column of a matrix)
    K[:, 0:ghost_len, :] = V_mat[0:ghost_len, :] * \
                                ( -c_mat[0:ghost_len, :]*dxU[:, 0:ghost_len, :]
                                -s_mat[0:ghost_len, :]*dyU[:, 0:ghost_len, :]
                                -U_mat[:, 0:ghost_len, :]/(2*r_mat[0:ghost_len, :]))

    # bottom
    K[:, -ghost_len:, :] = V_mat[-ghost_len:, :] *\
                                ( -c_mat[-ghost_len:, :]*dxU[:, -ghost_len:, :]
                                -s_mat[-ghost_len:, :]*dyU[:, -ghost_len:, :]
                                -U_mat[:, -ghost_len:, :]/(2*r_mat[-ghost_len:, :]))



    ################################
    # internal calculation of flux #
    ################################
    K[:, ghost_len:-ghost_len, ghost_len:-ghost_len] = -dxE[:, ghost_len:-ghost_len, ghost_len:-ghost_len] \
                                                       -dyF[:, ghost_len:-ghost_len, ghost_len:-ghost_len]

    return K


def simulate(U_mat, K_hist, D, ste_t, TF, dt, ghost_len, V_mat, s_mat, c_mat, r_mat, Mx):
    t = 0
    while t<TF:

        K_hist[1:,:] = K_hist[0:-1,:]
        K_curr = get_K(U_mat, D, ghost_len, V_mat, s_mat, c_mat, r_mat, Mx)
        K_hist[0] = K_curr
        U_mat += dt* np.tensordot(ste_t, K_hist, 1)

        t+=dt
    return U_mat

# Hardcoded function for this part to get good contoures
def denoise(mat):
    b = np.abs(mat)<0.00001
    mat[b] = 0

#############################################
# defining stencils for spatial derivatives #
#############################################
s_tim = np.array([-0.02084314277031176, 0.166705904414580469, -0.77088238051822552, 0,\
        0.77088238051822552, -0.166705904414580469, 0.02084314277031176])
s60 = np.array([-2.19228033900, 4.74861140100,-5.10885191500, 4.46156710400,
                -2.83349874100, 1.12832886100, -0.20387637100 ])
s51 = np.array([-0.20933762200, -1.08487567600, 2.14777605000, -1.38892832200,
                0.76894976600, -0.28181465000, 0.048230454000])
s42 = np.array([0.049041958000, -0.46884035700, -0.47476091400, 1.27327473700,
                -0.51848452600, 0.16613853300, -0.026369431000])
s_back = np.vstack([s60, s51, s42])


ste_t = [2.3025580888383, -2.4910075998482, 1.5743409331815, -0.3858914221716]
#############################################################################


if __name__ =="__main__":
    import time
    n =200
    dts = 0.0569
    TF =  dts*2000
    Mx = 0.5
    gh_len = 3
    D = get_D(n, s_tim, s_back, 7)
    K_hst = np.zeros((4,4,n,n))

    U_mat, x, y = in_U_s(n, n, 1, -n//2, -n//2, Mx)
    # it is better to calculate V(theta) once as it is invariant for a given node
    V, s, c , r= get_Vtheta(x, y, 0.5)

    atime = time.time()
    Uf = simulate(U_mat.copy(), K_hst, D, ste_t, TF, dts, gh_len, V, s, c, r, Mx)
    btime = time.time()
    print("time required: ", btime-atime)

    plt.figure()
    plt.title("initial condition for rho*")
    plt.contourf(x, y, U_mat[0])
    plt.colorbar()
    denoise(Uf[0])
    plt.figure()
    plt.title("rho* at t* = "+ str(TF))
    plt.contourf(x, y, Uf[0])
    plt.colorbar()

    plt.figure()
    plt.title("non dimensional speed at t* = "+str(TF))
    plt.contourf(x, y, np.sqrt(Uf[1]**2+Uf[2]**2))
    plt.colorbar()

    plt.show()
