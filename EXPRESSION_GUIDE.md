### MHR 仓库表情控制指南

本指南总结了 MHR 仓库中“表情（facial expression）”相关的核心代码位置和使用方式，方便后续查阅和修改。

---

### 1. 表情参数的基本概念

- **表情参数维度**
  - 身份（identity）blendshape 数量：`NUM_IDENTITY_BLENDSHAPES = 45`
  - 面部表情（facial expression）blendshape 数量：`NUM_FACE_EXPRESSION_BLENDSHAPES = 72`
  - 定义位置：`mhr/mhr.py`

- **表情参数张量**
  - 名称：`face_expr_coeffs`
  - 形状：`[batch_size, 72]`
  - 作用：控制面部表情对应的 72 维 blendshape 系数。

- **72 维“每一维代表什么”**
  - `mhr/mhr.py` 不会硬编码 72 个维度的语义；72 维的含义与顺序由模型资产（FBX/model）里 blendshape 的命名与排列决定。
  - 代码里约定：表情参数对应 `character.parameter_transform.names` **最后 72 个**条目（见 `set_blendshape_parameter_sets()`）。
  - 在集群/无 GUI 环境下，推荐用脚本直接导出名称列表（index → name）：

```bash
# 默认从 ./assets 读取
python tools/dump_expression_names.py --no-pose-correctives

# 指定资产目录（例如共享存储路径）
python tools/dump_expression_names.py --assets /path/to/assets --lod 1 --device cpu --no-pose-correctives

# 如果遇到 OpenMP 共享内存权限问题（某些容器/集群会禁用 shm）
export KMP_USE_SHM=0
python tools/dump_expression_names.py --no-pose-correctives
```

---

### 2. MHR 主模型中的表情控制

- **核心类**：`MHR`
  - 文件：`mhr/mhr.py`

- **前向调用接口**
  - 函数签名：
    - `forward(self, identity_coeffs, model_parameters, face_expr_coeffs, apply_correctives: bool = True)`
  - 关键逻辑：
    - 如果 `face_expr_coeffs is None`，会在内部用全零张量填充（中性表情）：
      - `face_expr_coeffs = torch.zeros(batch, NUM_FACE_EXPRESSION_BLENDSHAPES)`
    - 将 `identity_coeffs` 与 `face_expr_coeffs` 在列维度拼接：
      - `coeffs = torch.cat([identity_coeffs, face_expr_coeffs], dim=1)`
    - 将 `coeffs` 传入 `self.character_torch.blend_shape.forward(coeffs)`，从而驱动带表情的面部几何。

- **参数集划分（身份 vs 表情）**
  - 函数：`set_blendshape_parameter_sets(character: pym_geometry.Character)`
  - 作用：
    - 在 `character.parameter_transform` 中添加两个参数集合：
      - `"identity"`：对应身份 blendshape 的参数位置。
      - `"faceExpression"`：对应面部表情 blendshape 的参数位置。
  - 文件位置：`mhr/mhr.py` 末尾。

- **简单使用示例**
  - 文件：`demo.py`
  - 函数 `_prepare_input_data`（已改为**可控输入**）默认生成：
    - `identity_coeffs: [batch, 45]`（默认全 0）
    - `model_parameters: [batch, 204]`（默认全 0）
    - `face_expr_coeffs: [batch, 72]`（默认全 0 = 中性表情）
  - 如果你想沿用旧 demo 的“随机表情/随机输入”，传 `randomize=True` 即可。
  - 并调用：
    - `verts, skel_state = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)`

---

### 3. PyMomentum 拟合中的表情参数（C++ 求解器）

- 文件：`tools/mhr_smpl_conversion/pymomentum_fitting.py`
- 核心类：`PyMomentumModelFitting`

- **内部参数数量**
  - 从 MHR 模型中读取：
    - `_num_blendshapes = mhr_model.get_num_identity_blendshapes()`
    - `_num_expression_blendshapes = mhr_model.get_num_face_expression_blendshapes()`

- **拟合结果中的表情输出**
  - 方法：`get_fitting_results()`
  - 返回字典中包含：
    - `"face_expr_coeffs"`：从 `_solved_parameters` 最后 `_num_expression_blendshapes` 个元素中切片得到。

- **分阶段优化中的表情开关**
  - 辅助函数：`_add_stage(..., exclude_expression: bool = True, ...)`
  - 逻辑：
    - 通过 `expression_parameter_mask` 控制表情参数是否参与当前阶段优化：
      - `exclude_expression=True` 时，将表情参数 mask 置零（不优化表情）。
      - 否则激活表情参数，并在顶点权重中加入头部权重。
  - 在 `_create_hierarchical_masks` 中：
    - `Stage 1.0: face identity and expression`：通常锁定表情，只调身份。
    - `Stage 1.1: face expression`：在 `exclude_expression=False` 时，专门优化表情参数。

---

### 4. PyTorch 拟合中的表情参数（Adam 优化）

- 文件：`tools/mhr_smpl_conversion/pytorch_fitting.py`
- 相关类：`PyTorchMHRFitting`、`PyTorchSMPLFitting`

#### 4.1 MHR 侧表情优化（`PyTorchMHRFitting`）

- **可训练变量定义**
  - 函数：`_define_trainable_variables(...)`
  - 表情参数定义：
    - `num_face_expr = self._mhr_model.get_num_face_expression_blendshapes()`
    - `face_expr_coeffs = torch.zeros(num_frames, num_face_expr, device=self._device)`
    - 若 `exclude_expression=False` 则对 `face_expr_coeffs` 启用 `requires_grad_()`。

- **损失中的表情正则**
  - 函数：`_optimize_one_batch(...)`
  - 表情正则项：
    - `expression_regularization_loss = EXPRESSION_REGULARIZATION_WEIGHT * (...)`
    - 通过限制 `abs(face_expr_coeffs)` 不超过 `EXPRESSION_REGULARIZATION_THRESHOLD` 来抑制过大表情。
  - 常量：
    - `EXPRESSION_REGULARIZATION_WEIGHT = 1e4`
    - `EXPRESSION_REGULARIZATION_THRESHOLD = 0.3`

- **头部局部优化阶段**
  - 函数：`_get_head_optimization_config()`
  - 可优化参数列表中包含：
    - `"head_identity_coeffs"`、`"pose_params"`、`"face_expr_coeffs"` 等。
  - 函数：`_optimize_head_parameters(...)`
    - 在头部区域的顶点和边约束下，联合优化头部姿态、身份和表情参数。

- **前向调用时的表情使用**
  - 函数：`_get_mhr_vertices_batch(...)`
  - 每次调用 MHR 模型时，都会传入当前 batch 的 `face_expr_coeffs`：
    - `verts, _ = self._mhr_model(identity_coeffs, lbs_model_params, face_expr_coeffs, apply_correctives=True)`

#### 4.2 SMPL(X) 侧表情参数（`PyTorchSMPLFitting`）

- **SMPLX 表情变量**
  - 在 `_define_trainable_variables(...)` 中：
    - `expression = torch.zeros(num_frames, smpl_model.num_expression_coeffs, ...)`
    - 作为 `requires_grad=True` 的优化变量。

- **优化阶段中的使用**
  - 在 `_optimize_smpl(...)` 里：
    - `"expression"` 被加入 coarse 阶段的可优化参数列表，与 `global_orient`、`body_pose`、`betas` 等一起优化。

- **batch 参数准备**
  - 函数：`_get_batched_body_model_parameters(...)`
  - 确保对 SMPLX 调用时，`expression` 始终存在且 batch 维度正确。

---

### 5. 默认表情行为与扩展建议

- **默认行为**
  - 如果在调用 `MHR` 时不提供 `face_expr_coeffs` 或在 SMPLX 侧缺失 `expression`：
    - 框架会自动补零张量，即默认生成“中性表情”。

- **如果需要自定义表情控制，可以从以下入口入手：**
  - 直接控制：
    - 在应用层显式构造 `face_expr_coeffs` 并传入 `MHR(...)`，实现手动控制表情。
  - 调整拟合行为：
    - 修改 `pymomentum_fitting.py` 和 `pytorch_fitting.py` 中的：
      - `exclude_expression` 开关逻辑（是否参与拟合）。
      - 表情正则系数 `EXPRESSION_REGULARIZATION_WEIGHT` 和阈值 `EXPRESSION_REGULARIZATION_THRESHOLD`。
  - 调整维度：
    - 若更改表情 blendshape 数量，需要同时更新：
      - `NUM_FACE_EXPRESSION_BLENDSHAPES`
      - 资产文件（FBX / blendshape npz）
      - 与之相关的 mask、切片位置和下游工具。

