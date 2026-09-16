# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from flag_gems.fused.FLA.chunk import naive_chunk_gated_delta_rule_fwd


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.xfail(
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue"
)
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("V", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunk_gated_delta_rule_fwd_accuracy(B, T, H, K, V, dtype):
    device = flag_gems.device
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype)

    ref_o, ref_final_state = naive_chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state
    )

    result = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
    )
    # result is (g_cumsum, o, A, final_state, w_or_None, h_or_None, v_new_or_None)
    res_o = result[1]
    res_final_state = result[3]

    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(
        res_final_state.float(), ref_final_state, rtol=1.5, atol=1.0
    )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.xfail(
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue"
)
@pytest.mark.parametrize("T", [64, 128, 256])
def test_chunk_gated_delta_rule_fwd_no_initial_state(T):
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, K, V = 1, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    ref_o, _ = naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, None)

    result = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    res_o = result[1]

    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.parametrize("lengths", [(1, 1, 62), (3, 17, 44), (1, 63, 64)])
def test_chunk_gated_delta_rule_fwd_with_cu_seqlens(lengths):
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(1)

    B, T, H, K, V = 1, sum(lengths), 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    # Normalized keys keep the recurrence stable across multi-token sequences.
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = 0.1 * torch.randn(len(lengths), H, K, V, device=device, dtype=dtype)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.long)

    # The fixed-length reference is applied independently to each packed sequence.
    ref_outputs, ref_states = [], []
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        ref_o, ref_state = naive_chunk_gated_delta_rule_fwd(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale,
            initial_state[i : i + 1],
        )
        ref_outputs.append(ref_o)
        ref_states.append(ref_state)
    ref_o = torch.cat(ref_outputs, dim=1)
    ref_final_state = torch.cat(ref_states, dim=0)

    result = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    res_o = result[1]
    res_final_state = result[3]
    assert res_o.shape == (B, T, H, V)
    assert res_final_state.shape == (len(lengths), H, K, V)
    torch.testing.assert_close(res_o.float(), ref_o, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        res_final_state.float(), ref_final_state, rtol=2e-2, atol=2e-2
    )
