# 一键启动整合包方案（最小内容、本地离线）

## 目标
提供一个“解压即用”的本地整合包：用户一键启动后在网页选择图片，自动生成 `.ply`，再在同一站点打开 PlayCanvas Viewer 进行交互预览。模型权重默认在线下载，不打包 `.pt`。

## 用户流程
1. 运行 `start.bat` / `start.ps1`  
2. 浏览器自动打开 `http://localhost:7860`  
3. 通过文件选择器选择图片 → 点击“生成”  
4. 点击“打开 3D 预览”  
5. 新页面打开 Viewer 并加载 `.ply`

## 目录组织（最小）
```
bundle/
  start.bat
  start.ps1
  requirements.txt
  app/
    server.py
  sharp/                 # ml-sharp的代码
  viewer/                # PlayCanvas生成的 viewer 内容
    index.html
    index.js
    style.css
    static/...
  outputs/               # 生成的 .ply
```

## 运行机制
- **后端**：`app/server.py` 调用本地 `sharp` 代码生成 `.ply`  
- **前端**：简单上传页面 + 结果页面  
- **预览**：`viewer/` 下的 PlayCanvas Viewer 读取 `/outputs/*.ply`



## 关键实现说明
### 1) 一键启动脚本
- 创建 `.venv`（若不存在）  
- 检查本地环境
- 安装缺失依赖：`pip install -r requirements.txt`  
- 启动 `Server.py` 服务
- 自动打开浏览器

### 2) 推理调用
后端调用 `sharp predict` 或直接 import 运行推理逻辑，输出 `.ply` 到 `outputs/`。  
不加 `--render`(render需要安装vs 2022)，只生成 `.ply`。

### 3) Viewer 集成


## 依赖与模型下载策略
- 依赖由用户安装（`requirements.txt`）  
- 先检查本地目录是否存在模型文件。若不存在，运行启动脚本时自动从官方 URL 下载到本地目录，后续从本地加载。

## 注意事项
- Viewer 需要本地 HTTP 服务才能读取 `/outputs`  
- 不启用 `--render` 时不依赖 `gsplat` 编译环境  
