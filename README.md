# mini-infer

一个最小化的 C++ 推理框架骨架项目，用于快速搭建 `runtime + backend + tools + tests` 的工程结构。

当前项目重点是：

- 清晰的目录分层
- 可编译、可测试的最小代码
- 便于后续扩展 CPU / TensorRT / OpenVINO 后端

## 目录结构

```text
mini-infer/
  include/
    runtime/
      runtime.h
      tensor.h
      backend.h
  src/
    runtime/
    backends/
      cpu/
      tensorrt/
      openvino/   # optional
    graph/
    memory/
  models/
  benchmarks/
  tests/
  tools/
    infer_cli.cpp
  configs/
```

## 环境要求

- CMake >= 3.16
- 支持 C++17 的编译器（g++ / clang++）

## 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/infer_cli
```

传入模型路径（可选）：

```bash
./build/infer_cli models/demo.onnx
```

运行测试：

```bash
cd build
ctest --output-on-failure
```

## CMake 可选项

- `MINI_INFER_ENABLE_TENSORRT=ON/OFF`（默认 `ON`）
- `MINI_INFER_ENABLE_OPENVINO=ON/OFF`（默认 `OFF`）

示例：

```bash
cmake -S . -B build -DMINI_INFER_ENABLE_OPENVINO=ON
```

## 开发过程中常见问题与排查

### 1) 目录创建了但 git 看不到

**现象**：`models/`、`benchmarks/` 这种空目录没有被追踪。  
**原因**：Git 默认不追踪空目录。  
**解决**：放一个 `.gitkeep` 占位文件（项目已处理）。

### 2) `ctest` 提示 No tests were found

**现象**：在项目根目录执行 `ctest`，显示没有测试。  
**原因**：测试注册在 `build` 目录生成的 `CTestTestfile.cmake` 中。  
**解决**：

- 进入 `build/` 再跑：`cd build && ctest --output-on-failure`
- 或指定目录：`ctest --test-dir build --output-on-failure`

> 如果 `--test-dir` 方式在某些环境仍异常，优先用“进入 build 目录后执行”。

### 3) 编译通过但运行时提示找不到模型

**现象**：运行 `infer_cli` 时模型加载失败。  
**原因**：示例里只是记录路径，不会自动下载模型文件。  
**解决**：

- 把真实模型放到 `models/` 目录
- 或启动时传绝对/相对路径：`./build/infer_cli /path/to/model.onnx`

### 4) 后端代码写了但没有参与编译

**现象**：新增了后端 `.cpp`，运行行为未变化。  
**原因**：忘记在 `CMakeLists.txt` 的 `target_sources` 中添加文件，或对应开关没打开。  
**解决**：

- 检查 `add_library(mini_infer_core ...)` 是否包含该源文件
- 检查 `MINI_INFER_ENABLE_*` 选项是否开启
- 重新配置构建：`cmake -S . -B build`

### 5) 链接错误（undefined reference）

**现象**：链接阶段出现符号未定义。  
**原因**：声明和定义不一致，或源文件未编进目标。  
**解决**：

- 核对头文件声明与 `.cpp` 定义签名是否一致
- 确认相关 `.cpp` 在 `mini_infer_core` 中
- 清理后重建：删除 `build/` 后重新 `cmake -S . -B build`

### 6) `Tensor::numel()` 和 `data.size()` 不一致

**现象**：输出元素数和实际数据长度不一致。  
**原因**：`shape` 与 `data` 填充值不匹配，或者 shape 维度非法。  
**解决**：

- 构造输入时保证 `data.size() == shape` 各维乘积
- 在后续开发中增加 shape 校验和错误提示

### 7) 切换可选后端后行为不符合预期

**现象**：打开 OpenVINO/TensorRT 开关后仍是默认逻辑。  
**原因**：当前是骨架实现，后端 `run()` 仍返回输入透传。  
**解决**：后续补齐真实后端初始化、模型加载、执行流程。

### 8) 增加新模块后 include 报错

**现象**：`#include "runtime/xxx.h"` 找不到。  
**原因**：头文件路径与 `target_include_directories` 不一致。  
**解决**：

- 头文件统一放在 `include/` 下
- 保持 include 写法与目录结构一致
- 检查目标是否链接了 `mini_infer_core`

## 开发建议

- 每次新增模块先补一个 smoke test，保证最小可运行
- 在引入外部依赖（TensorRT/OpenVINO）前，先用 stub 对齐接口
- 优先保证 `runtime` API 稳定，再替换后端实现

## 下一步可扩展方向

- 增加 backend factory（按配置字符串创建后端）
- 支持读取 `configs/default.yaml`
- 增加更真实的 CPU 算子 demo（如 `add/relu/matmul`）
- 补充 benchmark 脚本与性能统计输出
