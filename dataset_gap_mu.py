import numpy as np

def sigmoid(epsilon,mu,d):
    s = 1/(1+np.exp((epsilon-mu)/d))

    return s

def F(x,epsilon,DOS,omega0,V0,n,d):
    delta = x[0]
    mu = x[1]

    dE = epsilon[1]-epsilon[0]
    t = sigmoid(np.abs(epsilon-mu),omega0,d)
    fac = 1/(2*np.sqrt((epsilon-mu)**2+delta**2*t**2))
    v2 = (1/2)*(1-((epsilon-mu)/np.sqrt((epsilon-mu)**2+delta**2*t**2)))

    F = np.array([
        1-dE*V0*np.sum(DOS*t**2*fac),
        n-dE*(DOS@v2)
    ])

    return F

def return_sol(epsilon,DOS,omega0,V0,n,d):
    sol = fsolve(
        F,
        [0.8,0.0],
        args=(epsilon,DOS,omega0,V0,n,d)
    )

    delta_sol = sol[0]
    mu_sol = sol[1]

    return delta_sol, mu_sol


n_values  = np.linspace(0.05, 1.0, 500)
V_values  = np.linspace(0.1, 5.0, 1000)
omega0    = np.linspace(0.1,2.0,800)
d = 0.002
