import importlib.metadata
import submodlib
import sklearn

for package in ["submodlib", "scikit-learn", "numpy"]:
    try:
        print(package, importlib.metadata.version(package))
    except Exception as exc:
        print(package, "version unavailable:", repr(exc))

print("submodlib import passed")
