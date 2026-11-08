# RapidDoc 镜像部署指南

镜像已推送至 [Docker Hub](https://hub.docker.com/r/hzkitty/rapid-doc)

## 镜像构建

如果需要自己构建镜像

Dockerfile 已移除本机 LibreOffice，六种 Office 格式（DOC/DOCX、XLS/XLSX、PPT/PPTX）由原生解析器处理；其他格式转 PDF 仍需外部 Gotenberg 服务。发布原生解析版本后，手动更新 Dockerfile 中的 `rapid-doc` 版本号，再构建镜像。

### 执行构建命令

```bash
cd docker

# 1. CPU 模式
docker build -f Dockerfile -t hzkitty/rapid-doc:0.9.11 .

# 2. GPU 模式
docker build -f DockerfileGPU -t hzkitty/rapid-doc:0.9.11-gpu .
```


## 运行部署

### 1. CPU 模式

仅CPU推理，资源占用较少：
```bash
docker-compose -f docker-compose.yml up -d
```
### 2. GPU 模式
```bash
docker-compose -f docker-compose-gpu.yml up -d
```

## 服务端口

- **8888**: RapidDoc Web API 服务端口

## API 使用

### 健康检查

```bash
curl http://localhost:8888/health
```

### 文档解析 API

```bash
# 上传文档进行解析
curl -X POST "http://localhost:8888/parse" \
     -F "file=@document.pdf" \
     -F "mode=pipeline"
```

## 配置文件详解

### .env 环境变量配置文件

`.env` 文件用于配置服务器和系统运行参数，支持以下配置项：

#### 基础配置

| 变量名                                | 默认值      | 说明                   |
|------------------------------------|----------|----------------------|
| `API_PORT`                         | `8888`   | RapidDoc Web API 端口  |
| `PADDLEOCRVL_VERSION`              |          | paddleocr-vl 版本      |
| `PADDLEOCRVL_VL_REC_BACKEND`       |          | paddleocr-vl backend |
| `PADDLEOCRVL_VL_VL_REC_SERVER_URL` |          | paddleocr-vl url     |


### 系统配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STARTUP_WAIT_TIME` | `15` | 启动等待时间（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `RAPID_MODELS_DIR` | `/app/models` | 模型文件存储目录 |

## 并发请求合并（单卡 A770）

`/file_parse` 不再让每个 HTTP 请求各自启动一条解析流水线。请求先进入内存队列，
dispatcher 在 `RAPID_DOC_API_BATCH_WINDOW_MS` 窗口内把**同配置**的请求合并成一次
RapidDoc batch；batch 内由 `RAPID_DOC_DOC_WORKERS` 个文档线程执行，多个请求从同一个
batch 输出目录读取自己的结果，batch 完成后统一清理。`clear_output_file=false`、
非 PDF、`return_model_output`、自定义模型等请求回退到独立执行。

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAPID_DOC_API_BATCH` | `1` | `0` 关闭合并，全部走独立执行 |
| `RAPID_DOC_API_BATCH_MAX_FILES` | `16` | 一次 batch 的最大文件数；超出会拆批 |
| `RAPID_DOC_API_BATCH_MAX_PAGES` | `1200` | 一次 batch 的最大页数；`0` = 不限 |
| `RAPID_DOC_API_BATCH_WINDOW_MS` | `100` | 首批请求后等待并发上传的窗口 |
| `RAPID_DOC_API_BATCH_MAX_WINDOW_MS` | `1000` | 有请求仍在上传时窗口的上限 |
| `RAPID_DOC_API_REQUEST_TIMEOUT_S` | `1800` | 单请求硬超时，超时返回 504 |
| `RAPID_DOC_API_QUEUE_MAX` | `64` | 等待队列上限，满时返回 503 |
| `RAPID_DOC_API_CONCURRENCY` | `1` | 保留的响应/队列线程数；实际解析由 batch 串行驱动 |

### 设备丢失自动重启

`libsycl_infer_onnx` 报 `UR_RESULT_ERROR_DEVICE_LOST`（Level Zero error 20）时，
进程的 Unified Runtime context 和全部队列已不可用，重建单个 session 无法恢复。
设备分配失败（`device allocation failed` / `UR_RESULT_ERROR_OUT_OF_RESOURCES`）
同样无法在进程内恢复：executor 已经在分配点回收空闲池并重试过一次，能抛到
Python 说明本进程常驻的模型加上本次调用的工作集已经放不进这张卡，后续调用只会
在同一份预算上继续失败。sycl-infer 层检测到这两类错误后都会立即以退出码 `75`
退出；`start_with_env.sh` 只在收到该退出码时重新启动 `app.py`，因此即使容器没有
`restart: always` 也能自动恢复。当前失败请求会返回 500，不保留进度。

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAPID_DOC_RESTART_ON_DEVICE_LOST` | `1` | `0` 关闭自动重启（改为抛出 `SyclDeviceLostError` / `SyclOutOfMemoryError`） |
| `RAPID_DOC_DEVICE_LOST_EXIT_CODE` | `75` | 触发 supervisor 重启的退出码 |

`RAPID_DOC_DOC_WORKERS=6` 和 layout 16 / det 32 / rec 128 / table 64 / page 24 保持不变。
**公式模型也走 sycl-infer**；`RAPID_DOC_FORMULA_ENGINE` 未设置时，layout / OCR /
table / formula 四个模型族都在同一张卡上执行。`SYCL_INFER_TRIM_MB=64` 控制每个
executor 在模型调用之间的空闲池上限；formula 的 Loop 在单次调用内部每达到
`PF_ONNX_LOOP_TRIM_MB`（默认 256 MiB）就回收一次自己的空闲池，所以 2 500 次
迭代的长公式也不会把设备池涨到 OOM。

### 实测（单张 A770，`error_pdfs` 12 份 / 957 页，formula=true）

| 场景 | 总 wall | 结果 |
|---|---:|---|
| 3 路并发，每路 4 个 PDF | **696.2 s** | 3/3 HTTP 200，12/12 文档 |
| 单个请求 12 个 PDF | **695.5 s** | 1/1 HTTP 200，12/12 文档 |
| 单个 300408（含公式） | **154.4 s** | HTTP 200，md 388 733 字节 |

同一构建内单请求与并发 batch 的同一文档 `md_content` md5 完全一致（12/12）；
无 500、无 OOM、无 device-lost。formula 的 sycl 输出与 ONNXRuntime 在同一输入上
逐元素一致（max|diff| = 0）。
