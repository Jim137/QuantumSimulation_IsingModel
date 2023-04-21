from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import QuantumCircuit
from qiskit.tools.monitor import job_monitor
import json


def azure_provider():
    "Returns an Azure Quantum provider instance."

    try:
        with open(".env/resource.json", "r") as f:
            resource = json.load(f)
    except FileNotFoundError:
        print("Cannot find resource.json. Please add a 'resource.json' to the folder '.env' under the root directory of the project.")
        raise

    provider = AzureQuantumProvider(
        resource_id=resource["id"],
        location=resource["location"],
    )
    return provider

def cost_estimation(circuit: QuantumCircuit, backend_name: str="quantinuum.qpu.h1-2", shots: int = 100):
    "Returns the cost estimation of the circuit on the specified backend."

    provider = azure_provider()
    backend = provider.get_backend(backend_name)
    cost = backend.estimate_cost(circuit, shots=100)
    return cost

def get_result(circuit: QuantumCircuit, backend_name: str="quantinuum.sim.h1-2sc", shots: int = 100):
    "Returns the result of the circuit on the specified backend."

    provider = azure_provider()
    backend = provider.get_backend(backend_name)
    job = backend.run(circuit, shots=shots)
    job_monitor(job)
    return job.result()

if __name__ == "__main__":
    "Test whether the provider is working and print the available targets."

    provider = azure_provider()
    print("This workspace's targets:")
    for backend in provider.backends():
        print("- " + backend.name())
