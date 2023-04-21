from azure.quantum.qiskit import AzureQuantumProvider
import json


def azure_provider():
    "Returns an Azure Quantum provider instance."

    with open(".env/resource.json", "r") as f:
        resource = json.load(f)

    provider = AzureQuantumProvider(
        resource_id=resource["id"],
        location=resource["location"],
    )
    return provider


if __name__ == "__main__":
    "Test whether the provider is working and print the available targets."

    provider = azure_provider()
    print("This workspace's targets:")
    for backend in provider.backends():
        print("- " + backend.name())
