/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "onnx/proto_utils.h"
#include "onnxsim.h"

namespace py = nanobind;
using namespace nanobind::literals;

struct PyModelExecutor : public ModelExecutor {
  using ModelExecutor::ModelExecutor;

  std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const override {
    std::vector<py::bytes> inputs_bytes;
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(inputs_bytes),
                   [](const onnx::TensorProto& x) {
                     const std::string str = x.SerializeAsString();
                     return py::bytes(str.data(), str.size());
                   });
    std::string model_str = model.SerializeAsString();
    auto output_bytes = _PyRun(py::bytes(model_str.data(), model_str.size()), inputs_bytes);
    std::vector<onnx::TensorProto> output_tps;
    std::transform(output_bytes.begin(), output_bytes.end(),
                   std::back_inserter(output_tps), [](const py::bytes& x) {
                     onnx::TensorProto tp;
                     tp.ParseFromString(std::string(x.c_str(), x.size()));
                     return tp;
                   });
    return output_tps;
  }

  virtual std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const = 0;
};

struct PyModelExecutorTrampoline : public PyModelExecutor {
  NB_TRAMPOLINE(PyModelExecutor, 1);

  /* Inherit the constructors */
  // using PyModelExecutor::PyModelExecutor;

  /* Trampoline (need one for each virtual function) */
  std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const override {
    NB_OVERRIDE_PURE_NAME(
        "Run", _PyRun, /* Name of function in C++ (must match Python name) */
        model_bytes, inputs_bytes /* Argument(s) */
    );
  }
};

NB_MODULE(onnxsim_cpp2py_export, m) {
  m.doc() = "ONNX Simplifier";

  using namespace py::literals;

  m.def("simplify",
        [](const py::bytes& model_proto_bytes,
           std::optional<std::vector<std::string>> skip_optimizers,
           bool constant_folding, bool shape_inference,
           size_t tensor_size_threshold) -> py::bytes {
          // force env initialization to register opset
          InitEnv();
          ONNX_NAMESPACE::ModelProto model;
          ParseProtoFromBytes(&model, model_proto_bytes.c_str(), model_proto_bytes.size());
          auto const result = Simplify(model, skip_optimizers, constant_folding,
                                       shape_inference, tensor_size_threshold);
          std::string out;
          result.SerializeToString(&out);
          return py::bytes(out.data(), out.size());
        }, "model_bytes"_a, "skip_optimizers"_a.none(),
        "constant_folding"_a = true, "shape_inference"_a = true, "tensor_size_threshold"_a)
      .def("simplify_path",
           [](const std::string& in_path, const std::string& out_path,
              std::optional<std::vector<std::string>> skip_optimizers,
              bool constant_folding, bool shape_inference,
              size_t tensor_size_threshold) -> bool {
             // force env initialization to register opset
             InitEnv();
             SimplifyPath(in_path, out_path, skip_optimizers, constant_folding,
                          shape_inference, tensor_size_threshold);
             return true;
           }, "in_path"_a, "out_path"_a,
           "skip_optimizers"_a.none(),
           "constant_folding"_a = true, "shape_inference"_a = true,
           "tensor_size_threshold"_a)
      .def("_set_model_executor",
           [](std::shared_ptr<PyModelExecutor> executor) {
             ModelExecutor::set_instance(std::move(executor));
           });

  // Release the C++ static reference to the Python model executor
  // This is called via atexit to prevent nanobind's "leaked instances" warning
  m.def("_clear_model_executor", []() {
    ModelExecutor::set_instance(nullptr);
  });

  py::class_<PyModelExecutor, PyModelExecutorTrampoline>(m, "ModelExecutor")
      .def(py::init<>())
      .def("Run", &PyModelExecutor::_PyRun);
}
