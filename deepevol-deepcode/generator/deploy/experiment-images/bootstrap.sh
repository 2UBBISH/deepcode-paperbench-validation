#!/usr/bin/env bash
# 把一台裸的阿里云 Ubuntu 实例烘成「实验 Agent 可直接用」的机器，
# 然后对它做 CreateImage 得到自定义镜像。见 docs/experiment-agent-design.md §8.3。
#
#   curl -fsSL .../bootstrap.sh | bash        # 或 scp 上去再跑
#
# 幂等：每一步都先探测再动手，重复跑安全。
#
# 装什么、以及为什么：
#   Docker                      rsa 在远端 Docker daemon 上建容器
#   NVIDIA Container Toolkit    GPU 要直通进容器，光有 Docker 不够
#   setupx-base:py310-proxy     rsa/setupx 的默认 base_image，上游没有 Dockerfile
#   rsa-grader:py311-v1         预烘后 _ensure_grader_image() 的 inspect 会短路，
#                               每台新机省掉一次构建
#   CloudMonitor Agent          现有硬约束：未预装的镜像不会被标记为可自动租赁
#                               （见 apps/api/.../remote_compute.py 的镜像门禁）
set -euo pipefail

log() { printf '\n\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

export DEBIAN_FRONTEND=noninteractive
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- apt 锁 ----
# ★ 刚开机的 Ubuntu 上 `unattended-upgrades` 正占着 dpkg 锁，
#   这时候 apt_get 会直接失败（exit=100，"Could not get lock
#   /var/lib/dpkg/lock-frontend ... held by process N (unattended-upgr)"）。
#   这是**间歇性**的：撞上了就整次烘镜像作废，撞不上就一切正常 ——
#   前几次成功纯属运气好。
#
#   用 apt 自己的锁等待（apt 2.0+ / Ubuntu 20.04+）而不是自己写轮询：
#   它覆盖 apt_get 的全部内部阶段，也覆盖 `ubuntu-drivers` 这类
#   在内部调 apt 的工具（靠 APT_CONFIG 传下去）。
APT_LOCK_WAIT=600
apt_get() { apt-get -o DPkg::Lock::Timeout="$APT_LOCK_WAIT" "$@"; }

# 让内部调用 apt 的工具（ubuntu-drivers）也继承这个等待。
printf 'DPkg::Lock::Timeout "%s";\n' "$APT_LOCK_WAIT" > /etc/apt/apt.conf.d/99deepevol-lock-wait

# 开机自动升级会和我们抢锁一路抢到底，烘镜像期间先停掉它。
# （只在本次构建里停；打出来的镜像照常保留该服务的启用状态。）
systemctl stop unattended-upgrades 2>/dev/null || true

# ------------------------------------------------------------------- Git ----
# 上传压缩包的路径要在这台机器上起一个只监听 127.0.0.1 的 git daemon，
# 把代码喂给 RSA 的容器（容器用 --network host，所以看得见回环）。
# 那需要宿主上有 git —— 基础镜像不一定带。
# Agent 侧 `git_daemon.serve_repo_on_machine` 也会幂等补装一次，
# 这里预装只是为了省掉每台新机器上那 10 来秒。
if have git; then
  log "git 已存在：$(git --version)"
else
  log "安装 git（zip 上传路径要用它在本机起 git daemon）"
  apt_get -qq update
  apt_get -qq install -y git
fi

# ---------------------------------------------------------------- Docker ----
if have docker; then
  log "Docker 已存在：$(docker --version)"
else
  log "安装 Docker（走阿里云源：download.docker.com 在国内机器上连接会被重置）"
  apt_get -qq update
  apt_get -qq install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg     # apt 以 _apt 用户跑，0600 的 keyring 读不到
  CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt_get -qq update
  apt_get -qq install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi

# registry mirror 只有境内机器需要。P1 验收实测：香港机器 pypi 10s 下 10MB、
# dockerhub 返回 401（鉴权挑战 = 通），而上海机器 pypi 8s 只下了 149KB、
# dockerhub 直接超时——差约 70 倍。境外机器配 mirror 反而多一层中转。
NEED_MIRROR=yes
if curl -sS -m 8 -o /dev/null https://registry-1.docker.io/v2/ 2>/dev/null; then
  NEED_MIRROR=no
fi
mkdir -p /etc/docker
if [ "$NEED_MIRROR" = no ]; then
  log "Docker Hub 直连可达（多半是境外机器），不配 registry mirror"
  [ -f /etc/docker/daemon.json ] || echo '{}' > /etc/docker/daemon.json
else
  log "配置 registry mirror（registry-1.docker.io 直连超时，多半是境内机器）"
  cat > /etc/docker/daemon.json <<'JSON'
{
  "registry-mirrors": [
    "https://dockerproxy.net",
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run"
  ]
}
JSON
fi

# ★ 这一步是 P0c 踩过的坑：`apt install docker-ce` 会**自动把 dockerd 起来**，
#   而 `systemctl enable --now docker` 对已经在运行的单元不会重启 —— 于是 daemon.json
#   根本没被加载。症状只表现为「拉不动镜像」，`docker info` 里连 Registry Mirrors
#   这一节都不会出现，很难联想到是配置没生效。必须显式 restart。
systemctl enable docker
systemctl restart docker
sleep 3
if [ "$NEED_MIRROR" = yes ]; then
  docker info 2>/dev/null | grep -A3 -i "registry mirrors" || {
    echo "!! daemon.json 没生效，registry mirror 缺失，后面拉镜像会失败" >&2
    exit 1
  }
fi

# ------------------------------------------------------- NVIDIA 驱动 -------
# ★ P1 验收踩到的坑：**不能用 `nvidia-smi` 是否存在来判断这是不是 GPU 机器**。
#   阿里云的基础公共镜像（acs:ubuntu_22_04_x64）**不含 NVIDIA 驱动**，
#   一台刚开出来的 T4 实例上 `nvidia-smi: command not found`——按旧写法会把
#   GPU 机器当成 CPU 机器静默跳过，等实验跑起来才发现没有 GPU。
#   硬件在不在要看 PCI 设备，驱动在不在才看 nvidia-smi。
have lspci || apt_get -qq install -y pciutils >/dev/null 2>&1 || true
GPU_HW=no
if lspci 2>/dev/null | grep -qi 'nvidia'; then GPU_HW=yes; fi
log "GPU 硬件: $GPU_HW / nvidia-smi: $(have nvidia-smi && echo 有 || echo 无)"

if [ "$GPU_HW" = yes ] && ! have nvidia-smi; then
  log "检测到 NVIDIA 硬件但没有驱动，安装中（几分钟）"
  apt_get -qq update
  # ubuntu-drivers 会按卡型挑推荐版本，比写死版本号稳。
  apt_get -qq install -y ubuntu-drivers-common
  ubuntu-drivers install --gpgpu 2>/dev/null || apt_get -qq install -y nvidia-driver-535-server
  if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    log "驱动就绪：$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  else
    echo "!! 驱动装了但 nvidia-smi 仍不可用，通常需要重启一次。" >&2
    echo "   烘镜像流程里：reboot 后重跑本脚本，再 CreateImage。" >&2
  fi
fi

# ------------------------------------------------ NVIDIA Container Toolkit --
if [ "$GPU_HW" = yes ] && have nvidia-smi; then
  if have nvidia-ctk; then
    log "NVIDIA Container Toolkit 已存在：$(nvidia-ctk --version | head -1)"
  else
    log "安装 NVIDIA Container Toolkit"
    # 三个坑叠在一起，每个的报错都不指向真因：
    #   1. 官方 GitHub Pages 源在国内机器上签名校验过不去 → 用 USTC 镜像
    #   2. 仓库签名 key 与 /gpgkey 那把**不是同一把**（NVIDIA 轮换过），
    #      直接 dearmor /gpgkey 会得到 NO_PUBKEY DDCAE044F796ECB0
    #   3. gpg --export 出来的 keyring 默认 0600，apt 以 _apt 用户读不到 → 必须 chmod a+r
    gpg --no-default-keyring --keyring /tmp/nvidia.gpg \
        --keyserver hkps://keyserver.ubuntu.com --recv-keys DDCAE044F796ECB0
    gpg --no-default-keyring --keyring /tmp/nvidia.gpg --export DDCAE044F796ECB0 \
        > /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    chmod a+r /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://mirrors.ustc.edu.cn/libnvidia-container/stable/deb/amd64 /" \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt_get -qq update
    apt_get -qq install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    sleep 3
  fi
  docker info 2>/dev/null | grep -i "runtimes" | grep -q nvidia \
    && log "GPU 直通就绪" \
    || echo "!! nvidia runtime 没注册上，容器里将拿不到 GPU" >&2
elif [ "$GPU_HW" = yes ]; then
  echo "!! 有 GPU 硬件但驱动不可用，跳过 Container Toolkit。容器里将拿不到 GPU。" >&2
else
  log "没有 NVIDIA 硬件（CPU 机器），跳过驱动与 Container Toolkit"
fi

# ---------------------------------------------------------------- 镜像 ------
log "预烘 rsa 需要的两个镜像"
docker build --pull -t setupx-base:py310-proxy -f "$SCRIPT_DIR/Dockerfile.setupx-base" "$SCRIPT_DIR"
docker build          -t rsa-grader:py311-v1   -f "$SCRIPT_DIR/Dockerfile.rsa-grader"   "$SCRIPT_DIR"

# ★ 验 grader 里的 pytest **真的能用**，不是「镜像建出来了」就算数。
#   实测踩过：pip 装到一份坏的 pytest 9.1.1 —— 模块是空的、元数据 pip 读不回来，
#   而 `import pytest` 不报错、`python -m pytest` 退出码 0 且无输出。
#   于是 rsa 收不到任何 test id，报成「pytest collected no tests」，
#   一句听起来完全是判据写错了的话。镜像建成功 ≠ 镜像可用。
if docker run --rm rsa-grader:py311-v1 python -m pytest --version >/dev/null 2>&1; then
  log "grader pytest 可用：$(docker run --rm rsa-grader:py311-v1 python -m pytest --version 2>&1 | head -1)"
else
  echo "!! grader 镜像里的 pytest 跑不起来。用它打出来的基础镜像会让每一次" >&2
  echo "   实验都停在「pytest collected no tests」，而那句话指向的是判据，不是这里。" >&2
  exit 1
fi

# ------------------------------------------------------------ CloudMonitor --
# API 侧的镜像门禁只把**预装了 CloudMonitor Agent** 的镜像标记为可自动租赁
# （docs/aliyun-gpu-lifecycle.md 里写作「现有硬约束」）。不装的话，
# 这台机器打出来的镜像永远不会出现在 preflight 的候选里 ——
# 而症状是「当前地域没有可用规格」，完全看不出是镜像没准备好。
#
# 安装脚本的 URL 带 `-internal`：那是**同地域内网**地址，
# 只有在该地域的 ECS 上才连得通。所以 REGION 必须与本机所在地域一致，
# 由 DEEPEVOL_ALIYUN_REGION 传入（默认取实例元数据）。
CMS_REGION="${DEEPEVOL_ALIYUN_REGION:-$(curl -sS -m 5 http://100.100.100.200/latest/meta-data/region-id 2>/dev/null || echo '')}"
CMS_VERSION="${DEEPEVOL_CMS_AGENT_VERSION:-2.1.55}"

if systemctl is-active --quiet cloudmonitor 2>/dev/null || [ -d /usr/local/cloudmonitor ]; then
  log "CloudMonitor Agent 已存在"
elif [ -n "$CMS_REGION" ]; then
  log "安装 CloudMonitor Agent（地域 $CMS_REGION，版本 $CMS_VERSION）"
  if REGION_ID="$CMS_REGION" VERSION="$CMS_VERSION" bash -c \
       "$(curl -sS -m 60 "https://cms-agent-${CMS_REGION}.oss-${CMS_REGION}-internal.aliyuncs.com/cms-go-agent/cms_go_agent_install.sh")"; then
    log "CloudMonitor Agent 装好了"
  else
    echo "!! CloudMonitor Agent 安装失败。打出来的镜像不会被标记为可自动租赁" >&2
  fi
else
  echo "!! 拿不到地域 id，跳过 CloudMonitor Agent 安装" >&2
fi

# ---------------------------------------------- CloudMonitor 开机自愈 -------
# ★ 实测（2026-08-24）：CmsGoAgent **不是 systemd 服务** —— 安装脚本把它作为
#   常驻进程直接拉起来，既没有 cloudmonitor.service，也没有任何开机自启。
#   于是「装进镜像」这件事本身是不成立的：用这张镜像克隆出来的新机器上，
#   /usr/local/cloudmonitor 的文件都在，但**没有任何东西会去启动它**，
#   阿里云那边一直是 not_installed，
#   remote_compute 的镜像门禁等满 360 秒后判开机失败。
#
#   症状离真因极远：用户看到的是「开机失败」，而真因是一个不会自启的进程。
#   实测重装一次约 2 分钟就能让阿里云报 running，所以这里装一个 oneshot：
#   开机时发现没跑就补装一次。跑着就立刻退出，不拖慢正常开机。
cat > /usr/local/bin/deepevol-cms-ensure.sh <<'ENSURE'
#!/usr/bin/env bash
# 镜像克隆出来的机器上补起 CloudMonitor Agent。见 bootstrap.sh 里的说明。
set -uo pipefail
if pgrep -f CmsGoAgent >/dev/null 2>&1; then
  echo "CmsGoAgent 已在运行，无需处理"; exit 0
fi
REGION="$(curl -sS -m 5 http://100.100.100.200/latest/meta-data/region-id 2>/dev/null || echo '')"
if [ -z "$REGION" ]; then
  echo "!! 拿不到地域 id，跳过" >&2; exit 0
fi
# 清掉镜像里那份（它记着烘镜像那台机器的状态），重装一份干净的。
rm -rf /usr/local/cloudmonitor
REGION_ID="$REGION" VERSION="${DEEPEVOL_CMS_AGENT_VERSION:-2.1.55}" bash -c \
  "$(curl -sS -m 90 "https://cms-agent-${REGION}.oss-${REGION}-internal.aliyuncs.com/cms-go-agent/cms_go_agent_install.sh")"
ENSURE
chmod +x /usr/local/bin/deepevol-cms-ensure.sh

cat > /etc/systemd/system/deepevol-cms-ensure.service <<'UNIT'
[Unit]
Description=DeepEvol: ensure Aliyun CloudMonitor agent is running (image clones start it dead)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/deepevol-cms-ensure.sh

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable deepevol-cms-ensure.service
log "已装开机自愈单元 deepevol-cms-ensure.service"

# 装完当场验一次。**验进程，不验目录** —— 旧写法只 `[ -d /usr/local/cloudmonitor ]`，
# 那个目录在「装了但根本不会启动」时同样存在，于是它一路放行，
# 把问题推迟到几周后的一句「当前地域没有可用规格 / 开机失败」。
# 目录存在证明不了任何事，进程在跑才算数。
if pgrep -f CmsGoAgent >/dev/null 2>&1; then
  log "CloudMonitor Agent 正在运行：$(pgrep -f CmsGoAgent | head -1)"
elif [ -d /usr/local/cloudmonitor ]; then
  # 文件在但没跑：开机自愈单元会在新机器上补起来，这里不算失败。
  log "CloudMonitor 文件就位但进程没跑 —— 依赖 deepevol-cms-ensure 在开机时补装"
else
  echo "!! 没检测到 CloudMonitor Agent。用这台机器打出来的镜像**不会**被标记为可自动租赁" >&2
  echo "   （见 remote_compute.py 的 _require_aliyun_preinstalled_cloudmonitor_image）。" >&2
  exit 1
fi

# ---------------------------------------------------------------- 自检 ------
log "自检"
docker --version
docker images --format '  {{.Repository}}:{{.Tag}} {{.Size}}'
have nvidia-smi && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
df -h / | tail -1

cat <<'EOF'

[bootstrap] 完成。接下来：
  1. 停机（阿里云按量实例要先停机才能 CreateImage）
  2. CreateImage 得到自定义镜像，把 image_id 填进 aliyun_gpu_options_json
  3. 之后每台新机开出来即用，省掉上面全部步骤 + grader 构建

注意驱动与 CUDA 的代际：`nvidia-smi` 顶部那个 "CUDA Version" 是**驱动支持的上限**。
驱动 470 → 只到 CUDA 11.4，装 cu12x/cu13x 的 torch 在容器里拿不到 GPU。
这正是 ComputeSpec 的 platform.min_compute_capability 要拦的东西。
EOF
