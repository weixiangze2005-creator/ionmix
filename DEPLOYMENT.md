# ionmix.cn 部署说明

本项目是 FastAPI 应用，不是纯静态网页。让 `ionmix.cn` 被任何人的电脑打开，需要完成两件事：

1. 把应用部署到云平台。
2. 把 `ionmix.cn` 的 DNS 解析到云平台。

## 推荐路线：Render + ionmix.cn

这是最快上线的路线。服务器在海外平台，通常不需要中国大陆 ICP 备案；但国内访问速度和稳定性不能和中国大陆服务器相比。

Render 官方 FastAPI 部署参数：

- Build Command：`pip install -r requirements.txt`
- Start Command：`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

本项目已经提供：

- `render.yaml`：Render Blueprint 配置。
- `Procfile`：通用 Web 启动配置。
- `/api/health`：部署平台健康检查接口。

## 上传前必须包含的运行文件

云端运行至少需要这些目录和文件：

```text
app/
data/processed/solvent_catalog.csv
data/solvent_seed.csv
data/lino3_solubility.csv
models/conductivity_model.joblib
models/lino3_solubility_model.joblib
models/training_report.json
models/lino3_solubility_report.json
requirements.txt
render.yaml
Procfile
```

注意：`models/*.joblib` 必须上传。否则云端没有模型，推荐结果会退化或报错。

## Render 操作步骤

1. 把本项目上传到 GitHub。
2. 打开 Render，选择 New -> Blueprint 或 New -> Web Service。
3. 连接 GitHub 仓库。
4. 如果选择 Web Service 手动配置，填写：
   - Runtime：Python
   - Build Command：`pip install --upgrade pip && pip install -r requirements.txt`
   - Start Command：`uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path：`/api/health`

免费实例内存只有 512MB。为了让免费档稳定运行，`render.yaml` 默认设置：

```text
IONMIX_CONDUCTIVITY_MODEL=disabled
```

这会在云端启动时跳过较大的 CALiSol-23 导电率森林模型，避免实例因内存超限被杀掉。网页仍会使用分子描述符、物理启发式规则和 LiNO3 小溶解度模型进行推荐。若升级到更高内存实例，可删除该环境变量或改为 `enabled`，重新部署后恢复导电率模型。
5. 部署完成后，Render 会先给一个 `*.onrender.com` 地址。
6. 先打开这个地址，确认网页和推荐接口正常。

## 绑定 ionmix.cn

1. 在 Render 对应服务里打开 Settings -> Custom Domains。
2. 添加：
   - `ionmix.cn`
   - `www.ionmix.cn`
3. Render 会显示需要添加的 DNS 记录。
4. 回到你购买 `ionmix.cn` 的域名控制台，添加 Render 要求的记录。

常见情况：

```text
主域名 ionmix.cn：
类型：ANAME / ALIAS / A
值：以 Render 页面实际显示为准

子域名 www.ionmix.cn：
类型：CNAME
值：你的 Render 服务地址，例如 ionmix.onrender.com
```

DNS 生效可能需要几分钟到数小时。生效后访问：

```text
https://ionmix.cn
https://www.ionmix.cn
```

## 如果改用中国大陆服务器

如果把网站部署到阿里云、腾讯云、华为云等中国大陆节点，通常需要 ICP 备案。备案与服务器接入商相关，不是单纯买了 `.cn` 域名就自动完成。

这种路线国内访问更稳，但上线慢，通常要先完成实名认证、购买大陆服务器、提交备案、等待审核。
