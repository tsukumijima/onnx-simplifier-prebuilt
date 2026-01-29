from onnxsim.onnx_simplifier import simplify, main

# register python executor
import atexit
import onnxsim.onnx_simplifier
import onnxsim.onnxsim_cpp2py_export
x = onnxsim.onnx_simplifier.PyModelExecutor()
onnxsim.onnxsim_cpp2py_export._set_model_executor(x)
del x  # C++ shared_ptr now owns the instance; no need to keep Python reference

# Ensure the C++ static reference is released before nanobind's module teardown
# to prevent "leaked instances/types" warnings at interpreter shutdown
atexit.register(onnxsim.onnxsim_cpp2py_export._clear_model_executor)

from .version import version as __version__  # noqa
