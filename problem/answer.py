import sys
from typing import Any
from time import time
import matplotlib.pyplot as plt

import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator

from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from quri_parts.algo.ansatz import HardwareEfficientReal
from quri_parts.algo.optimizer import Adam, OptimizerStatus, SPSA
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState, GeneralCircuitQuantumState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import create_proportional_shots_allocator
from quri_parts.circuit import UnboundParametricQuantumCircuit

from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms import MinimumEigensolver, VQEResult

sys.path.append("../")
from utils.challenge_2023 import ChallengeSampling, TimeExceededError

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

"""
####################################
add codes here
####################################
"""


def mansatz(l, n):
    ans = QuantumCircuit(n)
    
    if n == 4:
        ans.x(2)
        ans.x(3)
    elif n == 8:
        ans.x(4)
        ans.x(5)
        ans.x(6)
        ans.x(7)
    
    for j in range(l):
        for i in range(n):
            theta = Parameter('a'+str(j)+str(i))
            ans.ry(theta, i)
            
        for i in range(n-1):
            ans.cx(i, i+1)
            
    
    for i in range(n):
        beta = Parameter('b'+str(i))
        ans.rz(beta, i)
        
    for i in range(n):
        alpha = Parameter('c'+str(i))
        ans.ry(alpha, i)
        
    return ans

def mansatz_ZNE(l, n, par, k, s):
    
    if k%2 != 1:
        print('ERROR in ZNE')
        
    else:

        ans = QuantumCircuit(n)
        
        if n == 4:
            ans.x(2)
            ans.x(3)
        elif n == 8:
            ans.x(4)
            ans.x(5)
            ans.x(6)
            ans.x(7)

        if s > l:
            print('ERROR in FOLDING')

        for layer in range(l):
            
            if layer < s:
            
                for i in range(k):
                    for j in range(n):
                        if i%2 == 0:
                            ans.ry( par[layer*n+j], j)
                        else:
                            ans.ry( -par[layer*n+j], j)
                for j in range(n-1):
                    for i in range(k):
                        ans.cx(j, j+1)
            else:
                for j in range(n):
                    ans.ry( par[layer*n+j], j)
                for j in range(n-1):
                    ans.cx(j, j+1)

        for i in range(k):
            for j in range(n):
                if i %2 == 0:
                    ans.rz(par[-2*n+j], j)
                else:
                    ans.rz(-par[-2*n+j], j)

        for i in range(k):
            for j in range(n):
                if i %2 == 0:
                    ans.ry(par[-n+j], j)
                else:
                    ans.ry(-par[-n+j], j)

        return ans    

    
class CustomVQE_ZNE(MinimumEigensolver):
    
    def __init__(self, estimator, estimatorZNE, circuit, optimizer, n_qubits, callback=None):
        self._estimator = estimator
        self._estimatorZNE = estimatorZNE
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self.n_qubits = n_qubits
        
    def compute_minimum_eigenvalue(self, operators, aux_operators=None):
                
        # Define objective function to classically minimize over
        def objective(x):
            # Execute job with estimator primitive
            #job = self._estimator.run([self._circuit], [operators], [x])

            #x_axis = [1,1.7,3]
            x_axis = [1,1.5]
            aux_ZNE = []
            aux_n = 0
            
            for i in range(10):
                job = self._estimator(operators, self._circuit, [x])
                aux_n += job[0].value.real #est_result.values[0]
                
            aux_n = aux_n/10
            
            aux_ZNE.append(aux_n)
            
            aux_n = 0
            
            for i in range(10):
                est = self._estimatorZNE(operators, mansatz_ZNE(3,self.n_qubits,x, 3,1))
                aux_n += est.value.real
                
            aux_n = aux_n/10
            aux_ZNE.append(aux_n)
            
            model = np.poly1d(np.polyfit(x_axis, aux_ZNE, 1))
            value = model(0)

            # Save result information using callback function
            if self._callback is not None:
                self._callback(value, x)
            return value
            
        # Select an initial point for the ansatzs' parameters
        x0 = np.pi/4 * np.random.rand(self._circuit.num_parameters)
        
        # Run optimization
        try:
            res = self._optimizer.minimize(objective, x0=x0)
            result = VQEResult()
            result.cost_function_evals = res.nfev
            result.eigenvalue = res.fun
            result.optimal_parameters = res.x
            
        except TimeExceededError:
            result = 0
            print("Reached the limit of shots")
        
        # Populate VQE result

        return result

challenge_sampling = ChallengeSampling(noise=True)
    
class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> float:
        """
        ####################################
        add codes here
        ####################################
        """
        
        n_site = 4
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H",
            data_directory="../hamiltonian",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        pl = []

        data = []

        for pauli, coef in hamiltonian.items():

            aux = ['I']*n_qubits

            aux_p = str(pauli)

            if len(aux_p) == 1:
                pass
            elif len(aux_p) == 2:
                aux[int(aux_p[1])] = aux_p[0]
            else:
                auxl = aux_p.split(' ')

                for it in auxl:
                    aux[int(it[1])] = it[0]

            aux_f = ''
            for ij in aux:
                aux_f += str(ij)

            pl.append(aux_f)
            data.append(coef)
            
        ham = PauliSumOp(SparsePauliOp(pl,np.array(data)))
        spsa = SPSA(maxiter=1000)
        
        # Define a simple callback function
        intermediate_info = []
        def callback(value, x):
            intermediate_info.append([value,[x]])
        

        hardware_type = 'sc'
        shots_allocator = create_proportional_shots_allocator()
        measurement_factory = bitwise_commuting_pauli_measurement
        n_shots = 512
        
        sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
            n_shots, measurement_factory, shots_allocator, hardware_type))

        estimator = sampling_estimator

        ZNE_sampling_estimator = (
            challenge_sampling.create_sampling_estimator(
            n_shots, measurement_factory, shots_allocator, hardware_type))

        # Setup VQE algorithm
        custom_vqe = CustomVQE_ZNE(estimator, ZNE_sampling_estimator, mansatz(3,n_qubits), spsa,n_qubits, callback=callback)
        
        start = time()
        result = custom_vqe.compute_minimum_eigenvalue(hamiltonian)
        end = time()

        intermediate_info_aux = [intermediate_info[i][0] for i in range(len(intermediate_info))]
        
        aux_inter_info = intermediate_info_aux.copy()
        aux_ll = []
        for i in range(7):
            aux_ll.append(min(aux_inter_info))
            aux_inter_info.remove(min(aux_inter_info))
            
        average = 0
        for kk in aux_ll:
            average += kk

        average /=7
        
        print(f'execution time (s): {end - start:.2f}')
        
        return average


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
