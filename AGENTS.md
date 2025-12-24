# AGENTS.md

## 项目摘要
本仓库是 Apple SHARP 模型代码的本地化 Web 封装。核心机器学习实现位于 `src/sharp/`，预构建的 PlayCanvas 查看器位于 `viewer/`。
目前 `app/server.py` 为空，因此“Web 应用程序”尚未连通。

**核心使命**：将目前基于命令行的 SHARP 模型转化为一个本地运行的、基于浏览器的应用程序。

## 仓库布局 (事实来源)
- `src/sharp/`：SHARP 模型核心代码（CLI、模型架构、工具库）。
- `app/server.py`：Web 服务器的占位符（目前为空；注意文件名中的大写 `S`）。
- `viewer/`：预构建的 PlayCanvas 查看器资源（`index.js` 是压缩后的大型包，**请勿修改**）。
- `input/`, `output/`：用于存放输入/输出的本地文件夹（尚未连接到服务器）。
- `ONE_CLICK_BUNDLE_PLAN.md`：未来一键式 Web 包的设计笔记（尚未实现）。

## 当前状态与缺口分析

1.  **核心模型** (已就绪)：`src/sharp/` 可通过 CLI 运行，能稳定生成 `.ply` 文件。
2.  **3D 查看器** (已就绪)：`viewer/` 是编译好的应用，可渲染查看 `.ply` 文件。
3.  **Web 服务器** (缺失)：`app/server.py` 尚未实现。
4. **前端页面** (缺失)：需要首页 `page/index.html`，用于上传图片和展示结果。
4.  **环境配置** (注意)：存在 `requirements.txt`,目标是写一个自动脚本，在启动时检测是否创建虚拟环境和安装了所有依赖项，模型权重文件。如果缺失，脚本应自动安装。

## 待开发任务 (Immediate Action Items)

### 1. 实现 `app/server.py`
构建一个轻量级服务器（推荐 FastAPI 或 Flask）来编排整个流程：
- **静态资源服务**：必须通过 HTTP 服务托管 `viewer/` 目录（直接打开文件会导致浏览器跨域/文件访问限制）。
- **上传 API**：接收用户图片并保存至 `input/`。
- **预测 API**：调用 Python 逻辑处理图片。
    - **关键点**：建议直接 import `sharp.models` 或 `sharp.cli.predict` 以利用显存常驻，而不是使用 `subprocess`。
    - **性能**：模型应当在服务器启动时加载一次。
- **结果服务**：将生成的 `.ply` 文件路径（位于 `output/`）返回给前端。

### 2. 前端集成
`viewer/index.html` 目前仅加载默认场景。需要：
- 添加一个简单的 HTML 覆盖层（Overlay UI）用于上传图片。
- 接收服务器响应后，通过 JavaScript 动态更新 PlayCanvas 实例以加载新的 `.ply` URL。

### 3. 首页 `page/index.html`
- **功能**：提供上传图片的表单和展示结果的区域。
- **技术**：HTML5 表单、JavaScript 处理上传和结果展示。

## 技术细节与运行指南

### 运行模型 (CLI 与 调试)
如果 `pip install -e .` 失败（由于缺失打包元数据），请使用以下方式调用：
- **设置环境变量**：`PYTHONPATH=src`
- **手动调用示例 (Windows)**：
  `PYTHONPATH=src python -c "from sharp.cli import main_cli; main_cli()"`
- **主要命令**：
  - 预测：`sharp predict -i <images_or_dir> -o <output_dir>`
  - 渲染（仅限 CUDA）：`sharp render -i <ply_or_dir> -o <output_dir>`

### 模型检查点行为
- **默认行为**：代码定义了 `DEFAULT_MODEL_URL`。
- **本地硬编码路径**：代码中硬编码了一个 Windows 特定路径 (`src\models\sharp_2572gikvuh.pt`)。只有当该文件存在时才会使用，否则会通过 `torch.hub` 下载。
- **开发建议**：如果需要可移植的默认设置，请更新 `DEFAULT_CHECKPOINT_PATH` 为相对路径或环境变量。

## 开发边界与禁区 (Non-Goals)

- **不要修改 `src/sharp/models/`**：神经网络数学逻辑已完善，无需改动。
- **不要重构 `viewer/index.js`**：这是一个大型的压缩包。除非你要替换整个查看器构建流程，否则请将其视为二进制文件。
- **不要关注视频渲染**：Web UI 重点在于 **3D 交互预览**。`sharp render`（视频生成）目前仅作为 CLI 功能保留。
- **不要添加自动化测试**：目前仓库中没有任何自动化测试，不需要在此阶段引入。

## 代码修改位置建议
- CLI 和推理流程：`src/sharp/cli/predict.py`
- I/O 和高斯辅助函数：`src/sharp/utils/io.py`, `src/sharp/utils/gaussians.py`
- **Web 服务器实现**：`app/server.py`