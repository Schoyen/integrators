from setuptools import setup, find_packages

extras_require = {
    key: [f"jax[{key}]"] for key in ["cpu", "tpu", "cuda11_pip", "cuda12_pip", "cuda"]
}

setup(
    name="integrators",
    version="0.0.1",
    packages=find_packages(),
    extras_require=extras_require,
)
