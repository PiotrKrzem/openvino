// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"

#if defined(OPENVINO_ARCH_X86_64)

namespace ov {
namespace intel_cpu {
namespace kernel {
namespace random_uniform {

struct GeneratorCompileParams {
    element::Type out_data_type = element::f32;
};

struct PhiloxGeneratorCallArgs {
    void* dst_ptr;
    const void* key_ptr;
    const void* counter_ptr;
    const void* n_ptr;
    const void* min_ptr;
    const void* range_ptr;
    uint64_t work_amount = 0lu;
};

struct MersenneTwisterGeneratorCallArgs {
    void* dst_ptr;
    const void* state_ptr;
    const void* min_ptr;
    const void* range_ptr;
    uint64_t state_id = 0lu;
    uint64_t state_shift = 0lu;
    uint64_t step = 0lu;
    uint64_t work_amount = 0lu;
    uint64_t elements_remaining = 0lu;
    bool optimization_enabled = false;
    uint32_t out_data_type = 0u;

};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class PhiloxGenerator : public JitKernel<GeneratorCompileParams, PhiloxGeneratorCallArgs> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(PhiloxGenerator)

    explicit PhiloxGenerator(const GeneratorCompileParams& jcp);

    void generate() override;

private:
    using Vmm   = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Zmm,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Opmask,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;

    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_n_inc;
    RegistersPool::Reg<Xbyak::Reg64> r64_convert_0;
    RegistersPool::Reg<Xbyak::Reg64> r64_convert_1;
    RegistersPool::Reg<Xbyak::Reg64> r64_min;
    RegistersPool::Reg<Xbyak::Reg64> r64_f64_pow_52;

    const Xbyak::Reg64 r64_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // Vector registers.
    RegistersPool::Reg<Vmm> v_max_mul_n_64;
    RegistersPool::Reg<Vmm> v_max_mul_c_64;
    RegistersPool::Reg<Vmm> v_add_low_k;
    RegistersPool::Reg<Vmm> v_add_up_k;
    RegistersPool::Reg<Vmm> v_convert_0;
    RegistersPool::Reg<Vmm> v_convert_1;
    RegistersPool::Reg<Vmm> v_convert_2;
    RegistersPool::Reg<Vmm> v_n_inc;
    RegistersPool::Reg<Vmm> v_key_64;
    RegistersPool::Reg<Vmm> v_counter_64;
    RegistersPool::Reg<Vmm> v_n_64;
    RegistersPool::Reg<Vmm> v_min;
    RegistersPool::Reg<Vmm> v_range;
    RegistersPool::Reg<Vmm> v_res_perm;
    RegistersPool::Reg<Vmm> v_perm_16;

    void initVectors();

    void process();

    void runPhilox(const std::vector<Vmm>& vmm_res, const Vmm& vmm_key, const Vmm& vmm_counter, const Vmm& vmm_n);

    void calculateRound(const Vmm& vmm_k_0, const Vmm& vmm_k_1, const Vmm& vmm_c_0, const Vmm& vmm_c_1,
                        const Vmm& vmm_n_0, const Vmm& vmm_n_1, const Vmm& vmm_aux_0, const Vmm& vmm_aux_1);

    void raiseKey(const Vmm& vmm_k_0, const Vmm& vmm_k_1);

    void convert(const std::vector<Vmm>& vmm_dst, const std::vector<Vmm>& vmm_src);

    void tail(const std::vector<Vmm>& vmm_dst);

    static constexpr uint64_t ROUNDS_NUMBER = 10lu;
    static constexpr uint32_t CRUSH_RESISTANCE_CONST_LOWER_VALUE = 0x9E3779B9;
    static constexpr uint32_t CRUSH_RESISTANCE_CONST_UPPER_VALUE = 0xBB67AE85;
    static constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_N = 0xD2511F53;
    static constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER = 0xCD9E8D57;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class MersenneTwisterGenerator : public JitKernel<GeneratorCompileParams, MersenneTwisterGeneratorCallArgs> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(MersenneTwisterGenerator)

    explicit MersenneTwisterGenerator(const GeneratorCompileParams& jcp);

    void generate() override;

private:
    using Vmm   = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Zmm,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Opmask,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;

    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_state;
    RegistersPool::Reg<Xbyak::Reg64> r64_state_id;
    RegistersPool::Reg<Xbyak::Reg64> r64_state_shift;
    RegistersPool::Reg<Xbyak::Reg64> r64_step;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_elements_remaining;
    RegistersPool::Reg<Xbyak::Reg64> r64_optimization_enabled;
    RegistersPool::Reg<Xbyak::Reg64> r64_output_type;



    const Xbyak::Reg64 r64_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // Vector registers for input storage.
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_state;
    RegistersPool::Reg<Vmm> v_min;
    RegistersPool::Reg<Vmm> v_range;

    // Vector registers for generation.
    RegistersPool::Reg<Vmm> v_result;
    RegistersPool::Reg<Vmm> v_result_bitshift_11;
    RegistersPool::Reg<Vmm> v_result_bitshift_7;
    RegistersPool::Reg<Vmm> v_result_bitshift_7_const_1;
    RegistersPool::Reg<Vmm> v_result_bitshift_15;
    RegistersPool::Reg<Vmm> v_result_bitshift_15_const_2;
    RegistersPool::Reg<Vmm> v_result_bitshift_18;

    RegistersPool::Reg<Vmm> v_const_1;
    RegistersPool::Reg<Vmm> v_const_2;

    //Vector registers for conversion.
    RegistersPool::Reg<Vmm> v_mask;
    RegistersPool::Reg<Vmm> v_divisor;


    void initVectors();

    void process();

    void generateRandomNumbers(const Vmm& v_dst_0, const Vmm& v_dst_1);

    void convertToOutputTypeMersenne(const Vmm& v_result, const Vmm& v_min, const Vmm& v_range, const Vmm& v_dst, const Xbyak::Reg64& r64_elements_remaining);

    // Mersenne Twister constants
    static constexpr uint32_t MT_CONST_1 = 0x9D2C5680;
    static constexpr uint32_t MT_CONST_2 = 0xEFC60000;
    static constexpr uint32_t MT_N = 624;
    static constexpr uint32_t MT_M = 397;
    static constexpr uint32_t MT_U = 11;
    static constexpr uint32_t MT_S = 7;
    static constexpr uint32_t MT_T = 15;
    static constexpr uint32_t MT_L = 18;
    static constexpr uint32_t MT_4_ELEMENTS = 4;
    static constexpr uint32_t MT_2_ELEMENTS = 2;

    static constexpr uint32_t FLOAT_AS_VALUE = 0;
    static constexpr uint32_t FLOAT16_AS_VALUE = 1;
    static constexpr uint32_t BFLOAT16_AS_VALUE = 2;
    static constexpr uint32_t INT_AS_VALUE = 3;
    static constexpr uint32_t INT64_AS_VALUE = 4;


};

}   // namespace random_uniform
}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov

#endif // OPENVINO_ARCH_X86_64
