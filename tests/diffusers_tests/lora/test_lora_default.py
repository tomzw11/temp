# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import math
import unittest

import numpy as np

import mindspore as ms
from mindspore import mint, nn

from mindone.peft import LoraConfig, get_peft_model
from mindone.peft.tuners.lora.layer import DEFAULT_LORA, Linear, copy_parameters


class TestDefaultLoraMechanism(unittest.TestCase):
    """测试 DEFAULT_LORA 占位层机制：静态图下通过参数拷贝实现 LoRA 热切换。"""

    def setUp(self):
        """每个测试前重置为 PyNative 模式。"""
        ms.set_context(mode=ms.PYNATIVE_MODE)

    def tearDown(self):
        """每个测试后恢复为 PyNative 模式。"""
        ms.set_context(mode=ms.PYNATIVE_MODE)

    # ========== copy_parameters 基础测试 ==========

    def test_copy_parameters(self):
        """测试 copy_parameters 能否正确拷贝参数。"""
        src = nn.Dense(4, 4)
        dst = nn.Dense(4, 4)

        # 设置为不同值
        src.weight.set_data(mint.ones((4, 4)))
        dst.weight.set_data(mint.zeros((4, 4)))

        copy_parameters(src, dst)

        # dst 应该与 src 一致
        np.testing.assert_array_almost_equal(src.weight.asnumpy(), dst.weight.asnumpy())
        # 修改 src 不影响 dst（拷贝而非引用）
        src.weight.set_data(mint.ones((4, 4)) * 2)
        self.assertAlmostEqual(dst.weight.asnumpy().sum(), 4.0, delta=0.01)

    # ========== DEFAULT_LORA 创建测试 ==========

    def test_default_lora_created_on_update_layer(self):
        """测试 update_layer 首次调用时，自动创建 DEFAULT_LORA 占位层。"""
        base_layer = nn.Dense(10, 10)
        lora_layer = Linear(base_layer, "lora_0", r=4, lora_alpha=8, lora_dropout=0.0)

        self.assertIn(DEFAULT_LORA, lora_layer.lora_A)
        self.assertIn(DEFAULT_LORA, lora_layer.lora_B)
        self.assertIn(DEFAULT_LORA, lora_layer.lora_dropout)
        self.assertIn(DEFAULT_LORA, lora_layer.scaling)

    def test_default_lora_not_duplicated(self):
        """测试多次调用 update_layer 不会重复创建 DEFAULT_LORA。"""
        base_layer = nn.Dense(10, 10)
        lora_layer = Linear(base_layer, "lora_0", r=4, lora_alpha=8, lora_dropout=0.0)

        count_before = len(lora_layer.lora_A)
        lora_layer.update_layer("lora_1", r=4, lora_alpha=8, lora_dropout=0.0, init_lora_weights=True, use_rslora=False)
        # 新增 lora_1，但 DEFAULT_LORA 不应重复创建
        self.assertEqual(len(lora_layer.lora_A), count_before + 1)

    def test_default_lora_created_for_conv(self):
        """测试 Conv2d LoRA 层也会创建 DEFAULT_LORA。"""
        from mindone.peft.tuners.lora.layer import Conv2d as LoraConv2d

        base_layer = nn.Conv2d(3, 3, 3, pad_mode="pad", padding=1, has_bias=False)
        lora_layer = LoraConv2d(base_layer, "lora_0", r=4, lora_alpha=8, lora_dropout=0.0)

        self.assertIn(DEFAULT_LORA, lora_layer.lora_A)
        self.assertIn(DEFAULT_LORA, lora_layer.lora_B)

    # ========== set_adapter 行为测试 ==========

    def test_set_adapter_pynative_mode(self):
        """PyNative 模式下 set_adapter 走原始逻辑，不拷贝到 DEFAULT_LORA。"""
        ms.set_context(mode=ms.PYNATIVE_MODE)

        base_layer = nn.Dense(10, 10)
        lora_layer = Linear(base_layer, "lora_0", r=4, lora_alpha=8, lora_dropout=0.0)
        lora_layer.update_layer("lora_1", r=4, lora_alpha=8, lora_dropout=0.0, init_lora_weights=True, use_rslora=False)

        # 记录 DEFAULT_LORA 的初始权重
        init_weight = lora_layer.lora_A[DEFAULT_LORA].weight.asnumpy().copy()

        # PyNative 下 set_adapter("lora_1") 不应拷贝到 DEFAULT_LORA
        lora_layer.set_adapter("lora_1")
        self.assertEqual(lora_layer.active_adapters, ["lora_1"])

        # DEFAULT_LORA 权重应保持不变
        np.testing.assert_array_almost_equal(
            lora_layer.lora_A[DEFAULT_LORA].weight.asnumpy(), init_weight
        )

    def test_set_adapter_graph_mode(self):
        """GRAPH_MODE 下 set_adapter 将参数拷贝到 DEFAULT_LORA，并激活 DEFAULT_LORA。"""
        ms.set_context(mode=ms.GRAPH_MODE)

        base_layer = nn.Dense(10, 10)
        lora_layer = Linear(base_layer, "lora_0", r=4, lora_alpha=8, lora_dropout=0.0)
        lora_layer.update_layer("lora_1", r=4, lora_alpha=8, lora_dropout=0.0, init_lora_weights=True, use_rslora=False)

        # GRAPH_MODE 下 set_adapter 激活 DEFAULT_LORA
        lora_layer.set_adapter("lora_1")
        self.assertEqual(lora_layer.active_adapters, [DEFAULT_LORA])

        # DEFAULT_LORA 的参数应与 lora_1 一致
        np.testing.assert_array_almost_equal(
            lora_layer.lora_A[DEFAULT_LORA].weight.asnumpy(),
            lora_layer.lora_A["lora_1"].weight.asnumpy(),
        )
        np.testing.assert_array_almost_equal(
            lora_layer.lora_B[DEFAULT_LORA].weight.asnumpy(),
            lora_layer.lora_B["lora_1"].weight.asnumpy(),
        )
        self.assertAlmostEqual(
            lora_layer.scaling[DEFAULT_LORA], lora_layer.scaling["lora_1"], delta=1e-6
        )

    def test_set_adapter_graph_mode_switching(self):
        """GRAPH_MODE 下多次切换 adapter，DEFAULT_LORA 参数正确更新。"""
        ms.set_context(mode=ms.GRAPH_MODE)

        base_layer = nn.Dense(10, 10)
        lora_layer = Linear(base_layer, "lora_0", r=4, lora_alpha=8, lora_dropout=0.0)
        lora_layer.update_layer("lora_1", r=4, lora_alpha=8, lora_dropout=0.0, init_lora_weights=True, use_rslora=False)

        # 切换到 lora_0
        lora_layer.set_adapter("lora_0")
        np.testing.assert_array_almost_equal(
            lora_layer.lora_A[DEFAULT_LORA].weight.asnumpy(),
            lora_layer.lora_A["lora_0"].weight.asnumpy(),
        )

        # 切换到 lora_1
        lora_layer.set_adapter("lora_1")
        np.testing.assert_array_almost_equal(
            lora_layer.lora_A[DEFAULT_LORA].weight.asnumpy(),
            lora_layer.lora_A["lora_1"].weight.asnumpy(),
        )

        # 再次切回 lora_0
        lora_layer.set_adapter("lora_0")
        np.testing.assert_array_almost_equal(
            lora_layer.lora_A[DEFAULT_LORA].weight.asnumpy(),
            lora_layer.lora_A["lora_0"].weight.asnumpy(),
        )

    def test_set_adapter_graph_mode_no_default_lora(self):
        """GRAPH_MODE 下但 DEFAULT_LORA 不存在时，降级走原始逻辑。"""
        ms.set_context(mode=ms.GRAPH_MODE)

        base_layer = nn.Dense(10, 10)
        lora_layer = Linear(base_layer, "lora_0", r=4, lora_alpha=8, lora_dropout=0.0)

        # 手动删除 DEFAULT_LORA
        del lora_layer.lora_A[DEFAULT_LORA]
        del lora_layer.lora_B[DEFAULT_LORA]
        del lora_layer.lora_dropout[DEFAULT_LORA]
        del lora_layer.scaling[DEFAULT_LORA]

        lora_layer.set_adapter("lora_0")
        # 应走原始逻辑，激活 lora_0
        self.assertEqual(lora_layer.active_adapters, ["lora_0"])

    # ========== Forward 计算一致性测试 ==========

    def test_forward_pynative_vs_graph_mode(self):
        """PyNative 和 GRAPH_MODE 下 forward 结果应一致。"""
        base_layer = nn.Dense(4, 4)
        x = mint.randn(2, 4)

        # PyNative 模式
        ms.set_context(mode=ms.PYNATIVE_MODE)
        lora_pynative = Linear(base_layer, "lora_a", r=2, lora_alpha=4, lora_dropout=0.0)
        out_pynative = lora_pynative(x)

        # GRAPH_MODE 模式
        ms.set_context(mode=ms.GRAPH_MODE)
        base_layer2 = nn.Dense(4, 4)
        base_layer2.weight.set_data(base_layer.weight.copy())
        base_layer2.bias.set_data(base_layer.bias.copy())
        lora_graph = Linear(base_layer2, "lora_a", r=2, lora_alpha=4, lora_dropout=0.0)

        # 拷贝参数确保一致
        copy_parameters(lora_pynative.lora_A["lora_a"], lora_graph.lora_A["lora_a"])
        copy_parameters(lora_pynative.lora_B["lora_a"], lora_graph.lora_B["lora_a"])
        copy_parameters(lora_pynative.lora_A[DEFAULT_LORA], lora_graph.lora_A[DEFAULT_LORA])
        copy_parameters(lora_pynative.lora_B[DEFAULT_LORA], lora_graph.lora_B[DEFAULT_LORA])

        out_graph = lora_graph(x)

        np.testing.assert_array_almost_equal(out_pynative.asnumpy(), out_graph.asnumpy(), decimal=4)

    # ========== get_peft_model 集成测试 ==========

    def test_peft_model_with_default_lora(self):
        """测试通过 get_peft_model 注入 LoRA 后 DEFAULT_LORA 正常工作。"""
        model = nn.Dense(10, 10)
        config = LoraConfig(r=4, lora_alpha=8, target_modules=["Dense"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config)

        # 检查 linear 层包含 DEFAULT_LORA
        for name, module in peft_model.cells_and_names():
            if hasattr(module, "lora_A"):
                self.assertIn(DEFAULT_LORA, module.lora_A)
                self.assertIn(DEFAULT_LORA, module.lora_B)
                break
        else:
            self.fail("No LoRA layer found in peft_model")

    def test_peft_model_forward_after_switch(self):
        """测试 get_peft_model 注入后，切换 adapter 后 forward 仍正常。"""
        ms.set_context(mode=ms.GRAPH_MODE)

        model = nn.Dense(4, 4)
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["Dense"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config)

        x = mint.randn(2, 4)

        # 首次 forward
        out1 = peft_model(x)

        # 切换 adapter（在 GRAPH_MODE 下应拷贝到 DEFAULT_LORA）
        for name, module in peft_model.cells_and_names():
            if hasattr(module, "set_adapter"):
                module.set_adapter("default_0")
                break

        # 再次 forward 应正常（不触发重编译）
        out2 = peft_model(x)

        self.assertEqual(out1.shape, out2.shape)
        self.assertEqual(out1.dtype, out2.dtype)


if __name__ == "__main__":
    unittest.main()