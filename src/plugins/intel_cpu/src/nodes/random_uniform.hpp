// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <random>
#include "kernels/x64/random_uniform.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

// Following const values are taken from the original paper:
// https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
constexpr uint32_t CRUSH_RESISTANCE_CONST_LOWER_VALUE = 0x9E3779B9;
constexpr uint32_t CRUSH_RESISTANCE_CONST_UPPER_VALUE = 0xBB67AE85;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_N = 0xD2511F53;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER = 0xCD9E8D57;
constexpr uint64_t ROUNDS_NUMBER = 10llu;

// Following const values are taken from the original paper (used by PyTorch):
// https://dl.acm.org/doi/pdf/10.1145/272991.272995
constexpr int32_t MERSENNE_STATE_N = 624;
constexpr int32_t MERSENNE_STATE_M = 397;

class RandomUniform : public Node {
public:
    union OutputType {
        double   f64;
        float    f32;
        float16  f16;
        bfloat16 bf16;
        int64_t  i64;
        int32_t  i32;
        uint32_t u32;
        uint16_t u16;
    };

    RandomUniform(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;

    void initSupportedPrimitiveDescriptors() override;

    bool needPrepareParams() const override;

    void prepareParams() override;

    void execute(dnnl::stream strm) override;

    void executeDynamicImpl(dnnl::stream strm) override;

    bool isExecutable() const override;

    void createPrimitive() override;

    bool created() const override;

    bool canBeInPlace() const override { return false; }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    std::string getPrimitiveDescriptorType() const override;

protected:
    bool needShapeInfer() const override;

private:
    void evalRange();

    void initEdgeValues(OutputType& dst, const void* src, const element::Type& output_type);

    void prepareGeneratorKernel();

    enum PortIndex { SHAPE = 0, MIN_VAL, MAX_VAL };
    enum AlgorithmType { STL = 0, PHILOX, MERSENNE_TWISTER};

    bool m_const_inputs[3] = {false, false, false};

    ov::element::Type m_output_prc;
    uint64_t m_global_seed = 0lu;
    uint64_t m_op_seed = 0lu;
    std::pair<uint64_t, uint64_t> m_state {0lu, 0lu};

    VectorDims m_out_shape = {};
    uint64_t m_out_el_num = 1lu;
    OutputType m_min_val;
    OutputType m_max_val;
    OutputType m_range_val;
    AlgorithmType m_algo = PHILOX;

    /////////////////////////////////////////////////////////////////////////////////

    ///// PARALLELISM /////

    std::shared_ptr<kernel::JitKernelBase> m_jit_kernel;

    struct ThreadParams {
        uint64_t work_amount = 0lu;
        uint64_t dst_shift = 0lu;
        uint64_t state_shift = 0lu;
        uint64_t step = 0lu;
    };

    struct MersenneTwisterThreadParams {
        uint64_t elements_to_generate = 0lu;
        uint64_t dst_shift = 0lu;
        uint64_t state_shift = 0lu;
        uint64_t step = 0lu;
    };

    uint64_t m_threads_num = 0lu;
    std::vector<ThreadParams> m_thread_params;

    /////////////////////////////////////////////////////////////////////////////////

    ///// PHILOX /////

    // Determines how many sequence elements of RNG sequence are skipped between runs.
    // 256 is chosen for parity with Tensorflow.
    static constexpr uint64_t SKIP_CONST = 256lu;

    // Philox algorithm returns 4 elements of RNG sequence per each invocation
    static constexpr uint64_t PHILOX_GROUP_SIZE = 4lu;

    // Output elements number threshold to execute on one thread.
    static constexpr uint64_t PHILOX_PARALLEL_EXECUTION_THRESHOLD = 1000lu;

    // Used to parallelize state generation
    uint64_t m_skip_count = 0lu;

    void preparePhiloxParams();

    std::pair<uint64_t, uint64_t> computePhilox(void* out, size_t work_amount, const std::pair<uint64_t, uint64_t>& prev_state);

    /////////////////////////////////////////////////////////////////////////////////

    ///// MERSENNE TWISTER /////

    // Mersenne Twister algorithm standardized to return 4 elements of RNG sequence per each invocation
    static constexpr uint64_t MERSENNE_TWISTER_GROUP_SIZE = 4lu;

    // Output elements number threshold to execute on one thread.
    static constexpr uint64_t MERSENNE_TWISTER_PARALLEL_EXECUTION_THRESHOLD = 1000lu;

    // Each sub-run of Mersenne Twister generates 624-sized state of 32 bit numbers, no parallelization.
    // Then 4 of these numbers are consumed to generate output data, which can be parallelized.
    // Therefore, the maximum number of threads is 624 / 4 = 156
    static constexpr uint64_t MERSENNE_TWISTER_MAXIMUM_THREADS_THRESHOLD = 156lu;

    // PyTorch reduces the execution time when generating 64-bit numbers when the range is below max value of uint32_t
    bool m_mersenne_twister_optimization_enabled = false;

    // Number of random elements generated per thread.
    uint64_t m_elements_generated = 0lu;

    // Number of uint32s consumed to generate one output the requested type.
    uint64_t m_elements_consumed_per_one_output = 0lu;


    void prepareMersenneTwisterParams();

    void computeMersenneTwister(void* out, size_t work_amount);

    static constexpr uint32_t FLOAT_AS_VALUE = 0;
    static constexpr uint32_t FLOAT16_AS_VALUE = 1;
    static constexpr uint32_t BFLOAT16_AS_VALUE = 2;
    static constexpr uint32_t INT_AS_VALUE = 3;
    static constexpr uint32_t INT64_AS_VALUE = 4;

    /////////////////////////////////////////////////////////////////////////////////

    ///// STL /////

    std::default_random_engine m_generator;

    template <typename T, typename DISTR_TYPE>
    void generateData(DISTR_TYPE distribution, void* out, size_t work_amount);

    void computeStl(void* out, size_t work_amount);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
