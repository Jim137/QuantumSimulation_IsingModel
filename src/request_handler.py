from azure.quantum.qiskit import AzureQuantumProvider
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


if __name__ == "__main__":
    "Test whether the provider is working and print the available targets."

    provider = azure_provider()
    print("This workspace's targets:")
    for backend in provider.backends():
        print("- " + backend.name())
