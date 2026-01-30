try:
    # ruff: noqa: ANN001, ANN201, ERA001, N803, N806
    import triton
    import triton.language as tl


    @triton.jit
    def fused_add_round_kernel(
            x_ptr,
            output_ptr,  # contents will be added to the output
            seed,
            n_elements,
            EXPONENT_BIAS,
            MANTISSA_BITS,
            BLOCK_SIZE: tl.constexpr,
    ):
        """
        A kernel to upcast 8bit quantized weights to bfloat16 with stochastic rounding
        and add them to bfloat16 output weights. Might be used to upcast original model weights
        and to further add them to precalculated deltas coming from LoRAs.
        """
        # Get program ID and compute offsets
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        x = tl.load(x_ptr + offsets, mask=mask)
        rand_vals = tl.rand(seed, offsets) - 0.5

        x = tl.cast(x, tl.float16)
        delta = tl.load(output_ptr + offsets, mask=mask)
        delta = tl.cast(delta, tl.float16)
        x = x + delta

        x_bits = tl.cast(x, tl.int16, bitcast=True)

        # Calculate the exponent. Unbiased fp16 exponent is ((x_bits & 0x7C00) >> 10) - 15 for
        # normal numbers and -14 for subnormals.
        fp16_exponent_bits = (x_bits & 0x7C00) >> 10
        fp16_normals = fp16_exponent_bits > 0
        fp16_exponent = tl.where(fp16_normals, fp16_exponent_bits - 15, -14)

        # Add the target dtype's exponent bias and clamp to the target dtype's exponent range.
        exponent = fp16_exponent + EXPONENT_BIAS
        MAX_EXPONENT = 2 * EXPONENT_BIAS + 1
        exponent = tl.where(exponent > MAX_EXPONENT, MAX_EXPONENT, exponent)
        exponent = tl.where(exponent < 0, 0, exponent)

        # Normal ULP exponent, expressed as an fp16 exponent field:
        # (exponent - EXPONENT_BIAS - MANTISSA_BITS) + 15
        # Simplifies to: fp16_exponent - MANTISSA_BITS + 15
        # See https://en.wikipedia.org/wiki/Unit_in_the_last_place
        eps_exp = tl.maximum(0, tl.minimum(31, exponent - EXPONENT_BIAS - MANTISSA_BITS + 15))

        # Calculate epsilon in the target dtype
        eps_normal = tl.cast(tl.cast(eps_exp << 10, tl.int16), tl.float16, bitcast=True)

        # Subnormal ULP: 2^(1 - EXPONENT_BIAS - MANTISSA_BITS) ->
        # fp16 exponent bits: (1 - EXPONENT_BIAS - MANTISSA_BITS) + 15 =
        # 16 - EXPONENT_BIAS - MANTISSA_BITS
        eps_subnormal = tl.cast(tl.cast((16 - EXPONENT_BIAS - MANTISSA_BITS) << 10, tl.int16), tl.float16, bitcast=True)
        eps = tl.where(exponent > 0, eps_normal, eps_subnormal)

        # Apply zero mask to epsilon
        eps = tl.where(x == 0, 0.0, eps)

        # Apply stochastic rounding
        output = tl.cast(x + rand_vals * eps, tl.bfloat16)

        # Store the result
        tl.store(output_ptr + offsets, output, mask=mask)

except Exception:
    import torch

    def fused_add_round_kernel(
            x: torch.Tensor,
            output: torch.Tensor,
            seed: int,
            n_elements: int,  # Kept for signature compatibility, but unused
            EXPONENT_BIAS: int,
            MANTISSA_BITS: int,
            BLOCK_SIZE: int = None,  # Kept for signature compatibility, but unused
    ):
        """
        Native PyTorch implementation of the fused_add_round_kernel.

        This performs:
        1. Upcast 8-bit weights (x) to match output precision.
        2. Add output weights (deltas) to x.
        3. Calculate the epsilon (quantization noise step) based on the target
           Float8 parameters (EXPONENT_BIAS, MANTISSA_BITS).
        4. Apply stochastic rounding (add noise proportional to epsilon).
        5. Store back to output.
        """

        # 1. Setup Generators for stochastic rounding
        # We use a specific generator to respect the seed argument
        gen = torch.Generator(device=output.device).manual_seed(seed)

        # 2. Load and Cast to calculation precision (Float32 for safety, or Float16)
        # Using Float32 ensures high precision during the intermediate math
        val_x = x.to(torch.float32)
        val_delta = output.to(torch.float32)

        # x = x + delta
        val = val_x + val_delta

        # 3. Calculate Epsilon (The Stochastic Rounding Step)
        # The Triton kernel calculates epsilon based on the magnitude of 'val'
        # mapped onto the specific Float8 exponent grid.

        # Extract exponent: val = mantissa * 2^exp.
        # torch.frexp returns exp such that 0.5 <= |mantissa| < 1.0.
        # IEEE 754 log2(x) is (exp - 1).
        _, exp_obj = torch.frexp(val)
        unbiased_exp = exp_obj - 1

        # Map to target Float8 exponent space
        target_exp = unbiased_exp + EXPONENT_BIAS

        # Clamp exponent to target dtype range.
        # Max is standard formulation (2*Bias + 1).
        # Min is 1. Why 1? In the original Triton kernel, subnormals (exp <= 0)
        # utilize a constant epsilon calculated based on exponent=1 (the smallest normal).
        max_exponent = 2 * EXPONENT_BIAS + 1
        target_exp_clamped = torch.clamp(target_exp, min=1, max=max_exponent)

        # Calculate ULP exponent: E_target - BIAS - Mantissa_Bits
        eps_exponent = target_exp_clamped - EXPONENT_BIAS - MANTISSA_BITS

        # Convert exponent to actual epsilon value: 2^eps_exponent
        eps = torch.pow(2.0, eps_exponent.to(torch.float32))

        # Mask epsilon where value is exactly 0 (matches `tl.where(x == 0, 0.0, eps)`)
        eps = torch.where(val == 0, 0.0, eps)

        # 4. Generate Random Noise [-0.5, 0.5]
        rand_vals = torch.rand(val.shape, generator=gen, device=val.device) - 0.5

        # 5. Apply Stochastic Rounding
        # output = x + (noise * epsilon)
        result = val + (rand_vals * eps)

        # 6. Store Result
        # In-place update of the output tensor, cast to bfloat16
        output.copy_(result.to(torch.bfloat16))

        # No return value needed as operation is in-place on output_ptr/output