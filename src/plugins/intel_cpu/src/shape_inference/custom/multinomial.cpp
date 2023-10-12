// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include <openvino/op/multinomial.hpp>

namespace ov {
namespace intel_cpu {
namespace node {
using namespace InferenceEngine;
Result MultinomialShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                    const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const VectorDims& input_shape = input_shapes[0].get();
    const auto& num_samples_mem = data_dependency.at(0);

    size_t num_samples;
    if (num_samples_mem->getDesc().getPrecision() == Precision::I32) {
        num_samples = reinterpret_cast<const int32_t*>(num_samples_mem->getData())[0];
    } else {  // equals I64 precision
        num_samples = reinterpret_cast<const int64_t*>(num_samples_mem->getData())[0];
    }

    VectorDims dims{num_samples};

    if (input_shape.size() == 2) {
        dims.insert(dims.begin(), input_shape[0]);
    }

    return {{dims}, ShapeInferStatus::success};
}

ShapeInferPtr MultinomialShapeInferFactory::makeShapeInfer() const {
    if (const auto multinomial = ov::as_type_ptr<const ov::op::v13::Multinomial>(m_op)) {
        return std::make_shared<MultinomialShapeInfer>();
    } else {
        OPENVINO_THROW("Unexpected operation type in the Multinomial shape inference factory");
    }
}
}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
