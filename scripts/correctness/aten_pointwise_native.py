"""Actual ATen CUDA baselines for the contiguous scalar-operation cohort.

These are benchmark calls only. Compiler recognition is authorized by saved
imported-reference proofs, independently of this name-to-API table.
"""

UNARY = (
    'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'ceil',
    'cos', 'cosh', 'erf', 'erfc', 'exp', 'exp2', 'expm1', 'floor', 'frac',
    'lgamma', 'log', 'log1p', 'log2', 'log10', 'neg', 'reciprocal', 'rsqrt',
    'sigmoid', 'sign', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc',
)
BINARY = ('atan2', 'div', 'fmax', 'fmin', 'fmod', 'hypot', 'mul', 'nextafter', 'pow')
# ATen dispatches these APIs through its Jiterator CUDA implementation.
NATIVE_RUNTIME_LIBRARIES = {name: ['libnvrtc.so.12'] for name in ('erfc','exp2','lgamma')}
NATIVE = {name: f'at::{name}_out(t[1],t[0]);' for name in UNARY}
NATIVE.update({name: f'at::{name}_out(t[2],t[0],t[1]);' for name in BINARY})
NATIVE.update({
    'add': 't[1].add_(t[0]);',
    'relu': 'at::relu_out(t[1],t[0]);',
    'hardsigmoid': 'at::hardsigmoid_out(t[1],t[0]);',
    'hardswish': 'at::hardswish_out(t[1],t[0]);',
    'leaky_relu': 'at::leaky_relu_out(t[1],t[0],s[0]);',
    'hardtanh': 'at::hardtanh_out(t[1],t[0],s[0],s[1]);',
    'silu': 'at::silu_out(t[1],t[0]);',
    'silu_cpu': 'at::silu_out(t[1],t[0]);',
    'gelu': 'at::gelu_out(t[1],t[0],"tanh");',
    'gelu_cpu_tanh': 'at::gelu_out(t[1],t[0],"tanh");',
    'clamp': 'at::clamp_out(t[1],t[0],s[0],s[1]);',
    'clamp_scalar_cpu': 'at::clamp_out(t[1],t[0],s[0],s[1]);',
    'clamp_cpu': 'at::clamp_out(t[3],t[0],t[1],t[2]);',
    'mse_elementwise': 'at::mse_loss_out(t[2],t[0],t[1],at::Reduction::None);',
    # This extracted stage takes an arbitrary normalization, unlike the public
    # MSE API's reduction modes. Time the actual ATen sub/mul composition;
    # record both calls, without labeling it a single native MSE kernel.
    'mse_backward': 'at::sub_out(t[2],t[0],t[1]); t[2].mul_(s[0]);',
})
