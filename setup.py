from __future__ import annotations

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "calvano_market_cpp",
        ["src/market.cpp", "src/bindings.cpp"],
        include_dirs=["include"],
        cxx_std=17,
    )
]


setup(
    packages=["scripts", "neural", "experiments"],
    py_modules=["calvano_market", "calvano_qlearning"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
