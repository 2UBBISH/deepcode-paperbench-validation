# 实验 Agent 的远端镜像

租来的机器上必须先有这两个镜像，`rsa setup` 才跑得起来。P1 把自定义镜像烘出来时
应当把它们预置进去（见 `docs/experiment-agent-design.md` §8.3）。

| 镜像 | 来源 | 为什么要我们自己建 |
| --- | --- | --- |
| `setupx-base:py310-proxy` | 本目录 `Dockerfile.setupx-base` | **rsa 与 setupx 两个仓里都没有它的 Dockerfile**（作者本机构建、未提交），而 `rsa/remote_backend.py` 与 `setupx/src/environment_manager.py` 都把它当默认 `base_image`。后者找不到还会去 `images.pull()`，对本地名必然失败 |
| `rsa-grader:py311-v1` | 本目录 `Dockerfile.rsa-grader` | 内容与 `rsa/remote_backend.py:255-258` 的内联 Dockerfile **逐字一致**。`_ensure_grader_image()` 会先 `docker image inspect`，命中就跳过构建——预先建好并打同一 tag，每台新机就省掉一次构建 |

## 构建

```bash
docker build --pull -t setupx-base:py310-proxy -f Dockerfile.setupx-base .
docker build          -t rsa-grader:py311-v1   -f Dockerfile.rsa-grader   .
```

## 国内机器的两个坑（2026-08-23 在阿里云实测）

1. **Docker Hub 直连不通**（`registry-1.docker.io` 超时），必须配 registry mirror。
2. **`apt install docker-ce` 会自动把 dockerd 起来**，而 `systemctl enable --now docker`
   对已运行的单元不会重启。所以 `daemon.json` 必须在 **`systemctl restart docker`** 之后
   才生效——否则 `docker info` 里根本没有 Registry Mirrors 那一节，且报错只表现为
   "拉不动镜像"，很难联想到是配置没加载。
