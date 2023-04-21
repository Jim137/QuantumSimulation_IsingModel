from qiskit import *
from qiskit import QuantumCircuit
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from math import pi
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_bloch_multivector


def digit_sum(n):
    num_str = str(n)
    sum = 0
    for i in range(0, len(num_str)):
        sum += int(num_str[i])
    return sum

# CZ (Controlled-Z)
# control qubit: q0
# target qubit: q1
def CZ(qp,q0,q1):
    qp.h(q1)
    qp.cx(q0,q1)
    qp.h(q1)

# f-SWAP
# taking into account the one-directionality of CNOT gates in the available devices
def fSWAP(qp,q0,q1):
    qp.cx(q0,q1)
    qp.h(q0)
    qp.h(q1)
    qp.cx(q0,q1)
    qp.h(q0)
    qp.h(q1)
    qp.cx(q0,q1)
    CZ(qp,q0,q1)

# CH (Controlled-Haddamard)
# control qubit: q1
# target qubit: q0
def CH2(qp,q0,q1):
    qp.sdg(q0)
    qp.h(q0)
    qp.tdg(q0)
    qp.h(q0)
    qp.h(q1)
    qp.cx(q0,q1)
    qp.h(q0)
    qp.h(q1)
    qp.t(q0)
    qp.h(q0)
    qp.s(q0)

# Fourier transform gates
def F2(qp,q0,q1):
    qp.cx(q0,q1)
    CH2(qp,q0,q1)
    qp.cx(q0,q1)
    CZ(qp,q0,q1)

def F0(qp,q0,q1):
    F2(qp,q0,q1)

def F1(qp,q0,q1):
    F2(qp,q0,q1)
    qp.sdg(q0)

# ROTATIONAL GATES
def RZ(qp,th,q0):
    qp.p(-th,q0)
def RY(qp,th,q0):
    qp.u(th,0.,0.,q0)
def RX(qp,th,q0):
    qp.u(th,0.,pi,q0)

# CRX (Controlled-RX)
# control qubit: q0
# target qubit: q1
def CRX(qp,th,q0,q1):
    RZ(qp,pi/2.0,q1)
    RY(qp,th/2.0,q1)
    qp.cx(q0,q1)
    RY(qp,-th/2.0,q1)
    qp.cx(q0,q1)
    RZ(qp,-pi/2.0,q1)

# Bogoliubov B_1
def B(qp,thk,q0,q1):
    qp.x(q1)
    qp.cx(q1,q0)
    CRX(qp,thk,q0,q1)
    qp.cx(q1,q0)
    qp.x(q1)

# This circuit can be implemented in ibmqx5 using qubits (q0,q1,q2,q3)=(6,7,11,10)
# It can also be implemented between other qubits or in ibqmx2 and ibqmx4 using fermionic SWAPS
# For instance, the lines commented correspond to the implementations:
# ibmqx2 (q0,q1,q2,q3)=(4,2,0,1)
# ibmqx4 (q0,q1,q2,q3)=(3,2,1,0)
def Udisg(Udis,lam,q0,q1,q2,q3):
    k=1
    n=4
    th1=-np.arccos((lam-np.cos(2*pi*k/n))/np.sqrt((lam-np.cos(2*pi*k/n))**2+np.sin(2*pi*k/n)**2))
    B(Udis,th1,q0,q1)
    F1(Udis,q0,q1)
    F0(Udis,q2,q3)
    #fSWAP(Udis,q2,q1) # for ibmqx2
    #fSWAP(Udis,q1,q2) # for ibmqx4
    F0(Udis,q0,q2)
    F0(Udis,q1,q3)
    #fSWAP(Udis,q2,q1) # for ibmqx2
    #fSWAP(Udis,q1,q2) # for ibmqx4

def Initial(qc,lam,q0,q1,q2,q3):
    if lam <1:
        qc.x(q3)

def Ising(qc,ini,udis,mes,lam,q0,q1,q2,q3,c0,c1,c2,c3):
    Initial(ini,lam,q0,q1,q2,q3)
    Udisg(udis,lam,q0,q1,q2,q3)
    mes.measure([q0, q1, q2, q3], [c0, c1, c2, c3])
    qc.compose(ini, inplace=True)
    qc.barrier()
    qc.compose(udis, inplace=True)
    qc.barrier()
    qc.compose(mes, inplace=True)
    # qc.add_circuit("Ising",ini+udis+mes)


# use local simulator
shots = 1024
backend = Aer.get_backend('qasm_simulator')
mag_sim = []
for i in range(8):
    q = QuantumRegister(4, "q")
    c = ClassicalRegister(4, "c")
    Udis = QuantumCircuit(q, c)
    # Udis.draw("mpl")
    # plt.show()
    ini = QuantumCircuit(q, c)
    mes = QuantumCircuit(q, c)
    Isex = QuantumCircuit(q, c)
    # Isex.draw("mpl")
    # plt.show()

    lam = i*0.25
    Ising(Isex,ini, Udis ,mes,lam,q[0],q[1],q[2],q[3],c[0],c[1],c[2],c[3])
    # ini.draw("mpl")
    # plt.show()
    # Udis.draw("mpl")
    # plt.show()
    # mes.draw("mpl")
    # plt.show()
    if i == 0:
        Isex.draw("mpl")
        plt.savefig("./quantum_circuit_for_1D_4_spin_ising_model.png")
        plt.clf()
        # plt.show()

    job = execute(Isex, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()

    r1 = list(counts.keys())
    r2 = list(counts.values())
    M=0
    for j in range(0,len(r1)):
        M=M+(4-2*digit_sum(r1[j]))*r2[j]/shots
    print("$\lambda$: ",lam,", $<\sigma_{z}>$: ",M/4)
    mag_sim.append(M/4)

def exact(lam):
    if lam <1:
        return lam/(2*np.sqrt(1+lam**2))
    if lam >1:
        return 1/2+lam/(2*np.sqrt(1+lam**2))
    return None
vexact = np.vectorize(exact)
l=np.arange(0.0,2.0,0.01)
l1=np.arange(0.0,2.0,0.25)
plt.figure(figsize=(9,5))
plt.plot(l,vexact(l),'k',label='exact')
plt.plot(l1, mag_sim, 'bo',label='simulation')
plt.xlabel('$\lambda$')
plt.ylabel('$<\sigma_{z}>$')
plt.legend()
plt.title('Magnetization of the ground state of n=4 Ising spin chain')
plt.savefig("./Magnetization_of_the_ground_state_of_n=4_Ising_spin_chain.png")
plt.show()

def Initial_time(qc,t,lam,q0,q1,q2,q3):
    qc.u(np.arccos(lam/np.sqrt(1+lam**2)),pi/2.+4*t*np.sqrt(1+lam**2),0.,q0)
    qc.cx(q0,q1)

def Ising_time(qc,ini,udis,mes,lam,t,q0,q1,q2,q3,c0,c1,c2,c3):
    Initial_time(ini,t,lam,q0,q1,q2,q3)
    Udisg(udis,lam,q0,q1,q2,q3)
    mes.measure([q0, q1, q2, q3], [c0, c1, c2, c3])
    qc.compose(ini, inplace=True)
    qc.barrier()
    qc.compose(udis, inplace=True)
    qc.barrier()
    qc.compose(mes, inplace=True)

# use local simulator
shots = 1024
backend = Aer.get_backend('qasm_simulator')
# We compute the time evolution for lambda=0.5,0.9 and 1.8
nlam=3
magt_sim=[[] for _ in range(nlam)]
lam0=[0.5,0.9,1.8]
for j in range(nlam):
    lam=lam0[j]
    for i in range(9):
        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        Udis = QuantumCircuit(q, c)
        # Udis.draw("mpl")
        # plt.show()
        ini = QuantumCircuit(q, c)
        mes = QuantumCircuit(q, c)
        Isex_time = QuantumCircuit(q, c)

        t = i * 0.25

        Ising_time(Isex_time, ini, Udis, mes, lam, t, q[0], q[1], q[2], q[3], c[0], c[1], c[2], c[3])

        job = execute(Isex_time, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        r1 = list(counts.keys())
        r2 = list(counts.values())
        M = 0
        for k in range(0, len(r1)):
            M = M + (4 - 2 * digit_sum(r1[k])) * r2[k] / shots
        magt_sim[j].append(M / 4)


def exact_time(lam,tt):
    Mt=(1 + 2*lam**2 + np.cos(4*tt*np.sqrt(1 + lam**2)))/(2 + 2*lam**2)
    return Mt
vexact_t = np.vectorize(exact_time)
t=np.arange(0.0,2.0,0.01)
tt=np.arange(0.0,2.25,0.25)
plt.figure(figsize=(10,5))
plt.plot(t,vexact_t(0.5,t),'b',label='$\lambda=0.5$')
plt.plot(t,vexact_t(0.9,t),'r',label='$\lambda=0.9$')
plt.plot(t,vexact_t(1.8,t),'g',label='$\lambda=1.8$')
plt.plot(tt, magt_sim[0], 'b*',label='simulation')
plt.plot(tt, magt_sim[1], 'r*',label='simulation')
plt.plot(tt, magt_sim[2], 'g*',label='simulation')
plt.plot(tt, magt_sim[0], 'b--')
plt.plot(tt, magt_sim[1], 'r--')
plt.plot(tt, magt_sim[2], 'g--')
plt.xlabel('time')
plt.ylabel('$<\sigma_{z}>$')
plt.legend()
plt.title('Time evolution |↑↑↑↑> state')
plt.savefig("./Time_evolution_all_up_state.png")
plt.show()