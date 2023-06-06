"""This module contains some handy gate operations for the quantum circuit."""
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import *
import numpy as np
from qiskit.circuit.quantumregister import Qubit
from typing import Union



def CZ(circuit: QuantumCircuit, qubit0: int, qubit1: int):
    "Applies a CZ gate to the specified qubits."

    circuit.append(CZGate(), [qubit0, qubit1])


def fSWAP(circuit: QuantumCircuit, qubit0: Union[int, Qubit], qubit1: Union[int, Qubit]):
    "Applies a fSWAP gate to the specified qubits."

    if type(qubit0) is int:
        q0 = circuit.qubits[qubit0]
    elif type(qubit0) is Qubit:
        q0 = qubit0
    else:
        raise TypeError("only int or Qubit type accepted")
    if type(qubit1) is int:
        q1 = circuit.qubits[qubit1]
    elif type(qubit1) is Qubit:
        q1 = qubit1
    else:
        raise TypeError("only int or Qubit type accepted")

    circuit.cx(q0, q1)
    circuit.h(q0)
    circuit.h(q1)
    circuit.cx(q0, q1)
    circuit.h(q0)
    circuit.h(q1)
    circuit.cx(q0, q1)
    CZ(circuit, q0, q1)


def CH(circuit: QuantumCircuit, qubit0: int, qubit1: int):
    "Applies a Controlled_Haddamard gate to the specified qubits."

    circuit.append(CHGate(), [qubit0, qubit1])


def RZ(circuit: QuantumCircuit, qubit: int, angle: float):
    "Applies a RZ gate to the specified qubit."

    circuit.append(RZGate(angle), [qubit])


def RX(circuit: QuantumCircuit, qubit: int, angle: float):
    "Applies a RX gate to the specified qubit."

    circuit.append(RXGate(angle), [qubit])


def RY(circuit: QuantumCircuit, qubit: int, angle: float):
    "Applies a RY gate to the specified qubit."

    circuit.append(RYGate(angle), [qubit])


def CRX(circuit: QuantumCircuit, qubit0: int, qubit1: int, angle: float):
    "Applies a Controlled_RX gate to the specified qubits."

    circuit.append(CRXGate(angle), [qubit0, qubit1])


def B(circuit: QuantumCircuit, qubit0: Union[int, Qubit], qubit1: Union[int, Qubit], thk: float):
    "Applies a Bogoliubov gate to the specified qubits."

    if type(qubit0) is int:
        q0 = circuit.qubits[qubit0]
    elif type(qubit0) is Qubit:
        q0 = qubit0
    else:
        raise TypeError("only int or Qubit type accepted")
    if type(qubit1) is int:
        q1 = circuit.qubits[qubit1]
    elif type(qubit1) is Qubit:
        q1 = qubit1
    else:
        raise TypeError("only int or Qubit type accepted")

    circuit.x(q1)
    circuit.cx(q1, q0)
    CRX(circuit, q0, q1, thk)
    circuit.cx(q1, q0)
    circuit.x(q1)

# Fourier transform gates
def F2(qp,q0,q1):
    qp.cx(q0,q1)
    CH(qp,q1,q0)
    qp.cx(q0,q1)
    CZ(qp,q0,q1)

def F0(qp,q0,q1):
    F2(qp,q0,q1)

def F1(qp,q0,q1):
    F2(qp,q0,q1)
    qp.sdg(q0)

if __name__ == "__main__":
    "Test the module."

    from qiskit import ClassicalRegister
    import matplotlib.pyplot as plt
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)
    CZ(circuit, 0, 1)
    fSWAP(circuit, 0, 1)
    CH(circuit, 0, 1)
    RZ(circuit, 0, np.pi/2)
    RX(circuit, 0, np.pi/2)
    RY(circuit, 0, np.pi/2)
    CRX(circuit, 0, 1, np.pi/2)
    B(circuit, 0, 1, np.pi/2)
    circuit.draw(output="mpl")
    plt.show()
