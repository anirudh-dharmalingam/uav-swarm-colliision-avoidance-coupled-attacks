# syntax=docker/dockerfile:1.7

# Ubuntu 24.04 image digest observed from the native amd64 pull.
# Pinning the digest prevents the base image from silently changing.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

# ============================================================
# DroneLab Research Environment
#
# Native Ubuntu 24.04 host
#
# Robotics:
#   ROS 2 Jazzy
#   PX4-Autopilot v1.17.0
#   px4_msgs v1.17.0
#   Gazebo Harmonic
#   Micro XRCE-DDS Agent v2.4.3
#
# FANET / networking:
#   ns-3.46.1
#   OMNeT++ 6.4.0
#   INET 4.7.0
#
# Research:
#   Python 3.12
#   Scientific computing
#   OpenCV
#   PyTorch 2.10
#   PyTorch Geometric 2.7
#   RL / optimization
#   Jupyter / TensorBoard
#
# Design:
#   /opt      = installed toolchain
#   /workspace = mutable research source/data/results
#   PX4       = mutable because SITL builds/write artifacts
#
# Host ownership:
#   USERNAME/UID/GID are passed at build time.
# ============================================================

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
ARG ROS_DISTRO=jazzy

ARG PX4_REF=v1.17.0
ARG PX4_MSGS_REF=v1.17.0
ARG XRCE_AGENT_REF=v2.4.3

ARG NS3_VERSION=3.46.1
ARG NS3_SHA1=a02ce48bf2cb46ab8bcc0da10e47bfca1e13cdf3

ARG OMNETPP_VERSION=6.4.0
ARG OMNETPP_SHA256=2dd2cfbd388e50e843f5713716f065e878df7bb104b69f3435f740b37b21d459


ARG ROS_APT_SOURCE_VERSION=1.2.0

# Keep compilation parallelism below nproc so a 16-core / ~31 GB
# machine does not get hammered by simultaneous large C++ builds.
ARG BUILD_JOBS=8

# These MUST be passed from the native Ubuntu host.
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=${TZ} \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    ROS_DISTRO=${ROS_DISTRO} \
    ROS_DOMAIN_ID=42 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ROS_LOCALHOST_ONLY=0 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PX4_DIR=/opt/PX4-Autopilot \
    ROS_WS=/opt/ros2_ws \
    NS3_DIR=/opt/ns-${NS3_VERSION} \
    OMNETPP_HOME=/opt/omnetpp-${OMNETPP_VERSION} \
    INET_HOME=/opt/inet \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HOME=/home/${USERNAME} \
    DRONELAB_USER=${USERNAME} \
    DRONELAB_UID=${USER_UID} \
    DRONELAB_GID=${USER_GID} \
    MAKEFLAGS=-j${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS=-j${BUILD_JOBS}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ============================================================
# 1. BASE SYSTEM
# ============================================================

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg2 \
        lsb-release \
        software-properties-common \
        locales \
        sudo \
        gosu \
        git \
        git-lfs \
        wget \
        unzip \
        zip \
        rsync \
        file \
        tree \
        less \
        vim \
        nano \
        tmux \
        htop \
        procps \
        iproute2 \
        iputils-ping \
        net-tools \
        tcpdump \
        traceroute \
        iperf3 \
        openssh-client \
        socat \
        netcat-openbsd \
        psmisc \
        time \
        tzdata \
        jq \
        build-essential \
        cmake \
        ninja-build \
        ccache \
        gdb \
        lldb \
        clang \
        lld \
        gcc \
        g++ \
        make \
        pkg-config \
        diffutils \
        bison \
        flex \
        perl \
        sed \
        gawk \
        shellcheck \
        cppcheck \
        astyle \
        lcov \
        libssl-dev \
        libxml2-dev \
        libxml2-utils \
        zlib1g-dev \
        libdw-dev \
        libfontconfig1-dev \
        libeigen3-dev \
        libboost-all-dev \
        libsqlite3-dev \
        libgsl-dev \
        libopenmpi-dev \
        openmpi-bin \
        libopenblas-dev \
        liblapack-dev \
        qt6-base-dev \
        qt6-base-dev-tools \
        qmake6 \
        libqt6svg6 \
        qt6-wayland \
        libwebkit2gtk-4.1-0 \
        libopenscenegraph-dev \
        doxygen \
        graphviz \
        xdg-utils \
        mesa-utils \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libxcb-cursor0 \
        libxkbcommon-x11-0 \
        xvfb \
        python3 \
        python3-dev \
        python3-full \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel \
        python3-yaml \
    ; \
    locale-gen en_US.UTF-8; \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8; \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime; \
    echo "${TZ}" > /etc/timezone; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /root/.cache

# ============================================================
# 2. ROS 2 APT SOURCE
# ============================================================

RUN set -eux; \
    add-apt-repository universe; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl; \
    curl -fsSL \
      -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb"; \
    dpkg -i /tmp/ros2-apt-source.deb; \
    rm -f /tmp/ros2-apt-source.deb; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /root/.cache

# ============================================================
# 3. ROS DEVELOPMENT TOOLS
#
# IMPORTANT:
# These are installed only AFTER the ROS repository exists.
# ros-dev-tools is the supported ROS Jazzy developer-tool bundle.
# ============================================================

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ros-dev-tools; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /root/.cache

# ============================================================
# 4. PX4 SOURCE + OFFICIAL TOOLCHAIN
#
# PX4 v1.17 supports Ubuntu 24.04 and Gazebo Harmonic.
# We keep the PX4 source tree because this is a development/
# research image, not a runtime-only SITL image.
#
# --no-nuttx avoids unnecessary embedded cross-toolchains,
# since this image's primary target is SITL/research.
# ============================================================

WORKDIR /opt

RUN git clone \
      --recursive \
      --depth 1 \
      --branch "${PX4_REF}" \
      https://github.com/PX4/PX4-Autopilot.git \
      PX4-Autopilot

WORKDIR /opt/PX4-Autopilot

RUN git submodule update --init --recursive; \
    RUNS_IN_DOCKER=true bash ./Tools/setup/ubuntu.sh --no-nuttx

RUN set -eux; \
    git rev-parse HEAD > /opt/PX4-Autopilot.GIT_COMMIT; \
    git describe --tags --always --dirty > /opt/PX4-Autopilot.GIT_DESCRIBE; \
    rm -rf /root/.cache /root/.local/share/pip /tmp/*

# ============================================================
# 5. ROS 2 JAZZY + GAZEBO BRIDGE + RESEARCH ROS PACKAGES
#
# ros-jazzy-desktop already pulls the ROS base stack.
# Avoid installing ros-base twice.
# ============================================================

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-desktop \
        ros-${ROS_DISTRO}-ros-gz \
        ros-${ROS_DISTRO}-ros-gz-sim \
        ros-${ROS_DISTRO}-ros-gz-bridge \
        ros-${ROS_DISTRO}-ros-gz-interfaces \
        ros-${ROS_DISTRO}-gz-tools-vendor \
        ros-${ROS_DISTRO}-gz-sim-vendor \
        ros-${ROS_DISTRO}-tf2 \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-tf2-geometry-msgs \
        ros-${ROS_DISTRO}-geometry-msgs \
        ros-${ROS_DISTRO}-nav-msgs \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-std-msgs \
        ros-${ROS_DISTRO}-visualization-msgs \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-rqt \
        ros-${ROS_DISTRO}-rqt-common-plugins \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-robot-localization \
        ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-pcl-ros \
        ros-${ROS_DISTRO}-geographic-msgs \
        ros-${ROS_DISTRO}-xacro; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /root/.cache

# ============================================================
# 6. ROSDEP INITIALIZATION
# ============================================================

RUN set -eux; \
    rosdep init 2>/dev/null || true; \
    rosdep update; \
    rm -rf /tmp/* /root/.cache/pip

# ============================================================
# 7. PX4 ROS 2 MESSAGES
# ============================================================

RUN mkdir -p /opt/ros2_ws/src

WORKDIR /opt/ros2_ws/src

RUN git clone \
      --depth 1 \
      --branch "${PX4_MSGS_REF}" \
      https://github.com/PX4/px4_msgs.git

RUN git -C /opt/ros2_ws/src/px4_msgs rev-parse HEAD \
      > /opt/px4_msgs.GIT_COMMIT

WORKDIR /opt/ros2_ws

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install \
      --from-paths src \
      --ignore-src \
      --rosdistro ${ROS_DISTRO} \
      -r -y

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build \
      --symlink-install \
      --event-handlers console_direct+

# ============================================================
# 8. MICRO XRCE-DDS AGENT
# ============================================================

WORKDIR /opt

RUN git clone \
      --depth 1 \
      --branch "${XRCE_AGENT_REF}" \
      https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
      Micro-XRCE-DDS-Agent

RUN git -C /opt/Micro-XRCE-DDS-Agent rev-parse HEAD \
      > /opt/Micro-XRCE-DDS-Agent.GIT_COMMIT

WORKDIR /opt/Micro-XRCE-DDS-Agent

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local; \
    cmake --build build --parallel "${BUILD_JOBS}"; \
    cmake --install build; \
    ldconfig; \
    rm -rf build /root/.cache

# ============================================================
# 9. MAIN RESEARCH PYTHON ENVIRONMENT
# ============================================================

RUN python3 -m venv --upgrade-deps /opt/venv

RUN /opt/venv/bin/python -m pip install \
      numpy==2.3.4 \
      scipy==1.16.3 \
      pandas==2.3.3 \
      matplotlib==3.10.8 \
      seaborn==0.13.2 \
      plotly==6.5.2 \
      scikit-learn==1.8.0 \
      sympy==1.14.0 \
      pyyaml==6.0.3 \
      opencv-python==4.13.0.92 \
      pillow==12.1.1 \
      imageio==2.37.2 \
      networkx==3.6.1 \
      shapely==2.1.2 \
      tqdm==4.67.1 \
      psutil==7.2.2 \
      cppyy==3.5.0

RUN /opt/venv/bin/python -m pip install \
      cvxpy==1.7.5 \
      osqp==1.0.0 \
      clarabel==0.11.1

RUN /opt/venv/bin/python -m pip install \
      mavsdk==3.0.1 \
      pymavlink==2.4.49

RUN /opt/venv/bin/python -m pip install \
      torch==2.10.0 \
      torchvision==0.25.0 \
      torchaudio==2.10.0

RUN /opt/venv/bin/python -m pip install \
      torch-geometric==2.7.0

RUN /opt/venv/bin/python -m pip install \
      gymnasium==1.2.2 \
      stable-baselines3==2.7.0

# PyTorch 2.10 / torchvision 0.25 / torchaudio 2.10 are a matched set.


RUN /opt/venv/bin/python -m pip install \
      jupyterlab==4.5.5 \
      notebook==7.5.3 \
      tensorboard==2.20.0


RUN set -eux; \
    /opt/venv/bin/python -m pip check; \
    /opt/venv/bin/python -m pip freeze > /opt/research-pip-freeze.txt; \
    rm -rf /root/.cache /root/.cache/pip /tmp/*

# ============================================================
# 10. ns-3.46.1
#
# Official release checksum is verified before extraction.
# ns-3.46.1 release notes identify cppyy==3.5.0 as the known
# working Cppyy version on Linux.
# ============================================================

WORKDIR /opt

RUN set -eux; \
    wget -q \
      "https://www.nsnam.org/release/ns-${NS3_VERSION}.tar.bz2" \
      -O "/tmp/ns-${NS3_VERSION}.tar.bz2"; \
    echo "${NS3_SHA1}  /tmp/ns-${NS3_VERSION}.tar.bz2" | sha1sum -c -; \
    tar -xjf "/tmp/ns-${NS3_VERSION}.tar.bz2"; \
    rm -f "/tmp/ns-${NS3_VERSION}.tar.bz2"

WORKDIR /opt/ns-${NS3_VERSION}

RUN ./ns3 configure \
      --enable-examples \
      --enable-tests; \
    ./ns3 build; \
    ./ns3 --version > /opt/ns3.version 2>&1 || true; \
    rm -rf /root/.cache /tmp/*

# ============================================================
# 11. OMNeT++ 6.4.0
# ============================================================

WORKDIR /opt

RUN set -eux; \
    wget -q \
      "https://github.com/omnetpp/omnetpp/releases/download/omnetpp-${OMNETPP_VERSION}/omnetpp-${OMNETPP_VERSION}-linux-x86_64.tgz" \
      -O "/tmp/omnetpp-${OMNETPP_VERSION}.tgz"; \
    echo "${OMNETPP_SHA256}  /tmp/omnetpp-${OMNETPP_VERSION}.tgz" | sha256sum -c -; \
    tar -xzf "/tmp/omnetpp-${OMNETPP_VERSION}.tgz" -C /opt; \
    rm -f "/tmp/omnetpp-${OMNETPP_VERSION}.tgz"

WORKDIR /opt/omnetpp-${OMNETPP_VERSION}

RUN printf '%s\n' \
      'PREFER_CLANG=no' \
      'PREFER_LLD=no' \
      'WITH_QTENV=yes' \
      'WITH_OSG=yes' \
      > configure.user

RUN source ./setenv; \
    ./configure; \
    make -j"${BUILD_JOBS}"; \
    rm -rf /root/.cache /tmp/*

# ============================================================
# 12. OMNeT++ PYTHON ENVIRONMENT
# ============================================================

WORKDIR /opt/omnetpp-${OMNETPP_VERSION}

RUN python3 -m venv .venv --upgrade-deps

RUN ./.venv/bin/python -m pip install --no-cache-dir \
      -r python/requirements.txt; \
    rm -rf /root/.cache /tmp/*

# ============================================================
# 13. INET 4.7.0
# ============================================================
ARG INET_VERSION=4.7.0

WORKDIR /opt

RUN git clone \
      --depth 1 \
      --branch "v${INET_VERSION}" \
      https://github.com/inet-framework/inet.git \
      inet

RUN git -C /opt/inet rev-parse HEAD > /opt/inet.GIT_COMMIT

WORKDIR /opt/inet

RUN source /opt/omnetpp-${OMNETPP_VERSION}/setenv; \
    make -j"${BUILD_JOBS}"; \
    rm -rf /root/.cache /tmp/*

RUN /opt/venv/bin/python -m pip install \
    empy==3.3.4 \
    kconfiglib==14.1.0 \
    pyros-genmsg

# ============================================================
# 14. RESEARCH WORKSPACE
# ============================================================

RUN mkdir -p \
      /workspace/controllers \
      /workspace/autonomy \
      /workspace/network \
      /workspace/attacks \
      /workspace/worlds \
      /workspace/models \
      /workspace/config \
      /workspace/launch \
      /workspace/experiments \
      /workspace/results/metrics \
      /workspace/results/logs \
      /workspace/results/pcap \
      /workspace/scripts \
      /workspace/docs \
      /workspace/src \
      /workspace/ros2_ws \
      /workspace/datasets \
      /workspace/checkpoints \
      /workspace/manifests

# ============================================================
# 15. RESEARCH USER
#
# IMPORTANT:
# ubuntu:24.04 does NOT guarantee an "ubuntu" login user.
# Create the requested user explicitly instead of usermod'ing
# a possibly nonexistent account.
# ============================================================

RUN set -eux; \
    EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    if [ -n "${EXISTING_USER}" ] && [ "${EXISTING_USER}" != "${USERNAME}" ]; then \
        rm -rf "/home/${USERNAME}"; \
        usermod \
          --login "${USERNAME}" \
          --home "/home/${USERNAME}" \
          --move-home \
          "${EXISTING_USER}"; \
    elif ! getent passwd "${USERNAME}" >/dev/null; then \
        if ! getent group "${USER_GID}" >/dev/null; then \
            groupadd --gid "${USER_GID}" "${USERNAME}"; \
        fi; \
        useradd \
          --uid "${USER_UID}" \
          --gid "${USER_GID}" \
          --create-home \
          --shell /bin/bash \
          "${USERNAME}"; \
    fi; \
    usermod -aG sudo,dialout,video,plugdev "${USERNAME}"; \
    printf '%s\n' \
      "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
      > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

# ============================================================
# 16. OWNERSHIP
#
# Immutable toolchains remain root-owned.
# Mutable source/build trees are user-owned.
# ============================================================

RUN set -eux; \
    chown -R \
      "${USER_UID}:${USER_GID}" \
      "/home/${USERNAME}" \
      /workspace \
      /opt/PX4-Autopilot \
      /opt/ros2_ws \
      /opt/Micro-XRCE-DDS-Agent; \
    chmod -R u+rwX \
      "/home/${USERNAME}" \
      /workspace \
      /opt/PX4-Autopilot \
      /opt/ros2_ws \
      /opt/Micro-XRCE-DDS-Agent

# ============================================================
# 17. USER ENVIRONMENT
# ============================================================

RUN cat > "/home/${USERNAME}/.dronelab_env" <<'EOF'
# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# PX4 ROS 2 messages
if [[ -f /opt/ros2_ws/install/setup.bash ]]; then
    source /opt/ros2_ws/install/setup.bash
fi

# Main research Python environment
if [[ -f /opt/venv/bin/activate ]]; then
    source /opt/venv/bin/activate
fi

export PX4_DIR=/opt/PX4-Autopilot
export ROS_WS=/opt/ros2_ws
export NS3_DIR=/opt/ns-3.46.1
export OMNETPP_HOME=/opt/omnetpp-6.4.0
export INET_HOME=/opt/inet

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export TZ="${TZ:-Etc/UTC}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Intentionally NOT setting use_sim_time globally.
# Simulation nodes should opt in explicitly:
#   --ros-args -p use_sim_time:=true
EOF

RUN printf '%s\n' 'source ~/.dronelab_env' >> "/home/${USERNAME}/.bashrc"

# ============================================================
# 18. RESEARCH MANIFEST
# ============================================================

RUN cat > /opt/dronelab-manifest.txt <<EOF
DroneLab Research Environment

Base:
  Ubuntu 24.04
  image digest: sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ROS:
  ROS_DISTRO=${ROS_DISTRO}
  ros-apt-source=${ROS_APT_SOURCE_VERSION}

PX4:
  REF=${PX4_REF}
  COMMIT=$(cat /opt/PX4-Autopilot.GIT_COMMIT)

PX4 ROS 2 messages:
  REF=${PX4_MSGS_REF}
  COMMIT=$(cat /opt/px4_msgs.GIT_COMMIT)

Micro XRCE-DDS:
  REF=${XRCE_AGENT_REF}
  COMMIT=$(cat /opt/Micro-XRCE-DDS-Agent.GIT_COMMIT)

Gazebo:
  Harmonic

ns-3:
  VERSION=${NS3_VERSION}

OMNeT++:
  VERSION=${OMNETPP_VERSION}

INET:
  VERSION=${INET_VERSION}
  COMMIT=$(cat /opt/inet.GIT_COMMIT)

Python:
  /opt/venv
  freeze=/opt/research-pip-freeze.txt

Build:
  BUILD_JOBS=${BUILD_JOBS}

Identity:
  NAME=${USERNAME}
  UID=${USER_UID}
  GID=${USER_GID}

Clock policy:
  wall clock = host/kernel clock
  simulation clock = Gazebo /clock
  ROS sim time = explicit use_sim_time
EOF

# ============================================================
# 19. BUILD-TIME VALIDATION
# ============================================================

RUN cat > /usr/local/bin/dronelab-build-test <<'EOF'
#!/usr/bin/env bash
set -eo pipefail

RESEARCH_USER="${DRONELAB_USER}"
RESEARCH_UID="${DRONELAB_UID}"
RESEARCH_GID="${DRONELAB_GID}"

fail() {
    echo "[FAIL] $*" >&2
    exit 1
}

echo "===== DroneLab BUILD VALIDATION ====="

echo "[1] OS"
grep -E '^(NAME|VERSION_ID|VERSION_CODENAME)=' /etc/os-release

echo "[2] Identity"
id "${RESEARCH_USER}"
test "$(id -u "${RESEARCH_USER}")" = "${RESEARCH_UID}" || fail "UID mismatch"
test "$(id -g "${RESEARCH_USER}")" = "${RESEARCH_GID}" || fail "GID mismatch"

echo "[3] Python"
/opt/venv/bin/python --version
/opt/venv/bin/python -m pip check

echo "[4] ROS"
test -f /opt/ros/jazzy/setup.bash
source /opt/ros/jazzy/setup.bash
source /opt/ros2_ws/install/setup.bash
command -v ros2 >/dev/null || fail "ros2 missing"
command -v rosdep >/dev/null || fail "rosdep missing"
command -v colcon >/dev/null || fail "colcon missing"
command -v vcs >/dev/null || fail "vcs missing"
ros2 pkg prefix px4_msgs >/dev/null || fail "px4_msgs missing"

echo "[5] Gazebo"
command -v gz >/dev/null || fail "gz missing"
gz sim --version

echo "[6] PX4"
test -f /opt/PX4-Autopilot/Makefile
test -f /opt/PX4-Autopilot.GIT_COMMIT

echo "[7] XRCE-DDS"
command -v MicroXRCEAgent >/dev/null || fail "MicroXRCEAgent missing"

echo "[8] ns-3"
test -x /opt/ns-3.46.1/ns3

echo "[9] OMNeT++ / INET"
test -x /opt/omnetpp-6.4.0/bin/opp_run
test -f /opt/omnetpp-6.4.0/setenv
test -d /opt/inet

echo "[10] Provenance"
test -s /opt/research-pip-freeze.txt
test -s /opt/dronelab-manifest.txt

echo "[PASS] BUILD-TIME VALIDATION PASSED"
EOF

RUN chmod +x /usr/local/bin/dronelab-build-test

RUN /usr/local/bin/dronelab-build-test

# ============================================================
# 20. RUNTIME INTEGRATION TEST
#
# This is intentionally NOT executed during docker build.
# It needs runtime networking/processes and therefore runs from
# the finished container on the native Ubuntu host.
#
# Verifies:
#   - wall-clock sanity
#   - ROS 2
#   - Gazebo
#   - Gazebo -> ROS /clock
#   - Micro XRCE-DDS Agent
#   - PX4 SITL
#   - PX4 -> ROS 2 topic flow
#   - PX4 timesync_status
# ============================================================

RUN cat > /usr/local/bin/dronelab-integration-test <<'EOF'
#!/usr/bin/env bash
set -Eeo pipefail

TEST_ROOT="${TEST_ROOT:-/workspace/results/integration-test}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

mkdir -p "${TEST_ROOT}"
rm -f "${TEST_ROOT}"/*.log "${TEST_ROOT}"/*.txt

cleanup() {
    set +e
    [[ -n "${PX4_PID:-}" ]] && kill "${PX4_PID}" 2>/dev/null || true
    [[ -n "${BRIDGE_PID:-}" ]] && kill "${BRIDGE_PID}" 2>/dev/null || true
    [[ -n "${AGENT_PID:-}" ]] && kill "${AGENT_PID}" 2>/dev/null || true
    pkill -f "MicroXRCEAgent udp4 -p 8888" 2>/dev/null || true
    pkill -f "parameter_bridge.*clock" 2>/dev/null || true
    pkill -f "px4_sitl" 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    echo "[FAIL] $*" >&2
    echo "Artifacts: ${TEST_ROOT}" >&2
    exit 1
}

echo "===== DroneLab RUNTIME INTEGRATION TEST ====="

echo "[1/8] Wall clock"
date -u
python3 - <<'PY'
from datetime import datetime, timezone
print("Python UTC:", datetime.now(timezone.utc).isoformat())
PY

EPOCH_A="$(date +%s)"
EPOCH_B="$(python3 -c 'import time; print(int(time.time()))')"
DELTA=$((EPOCH_A - EPOCH_B))
DELTA=${DELTA#-}
(( DELTA <= 2 )) || fail "wall-clock/Python clock mismatch > 2 seconds"

echo "[2/8] ROS 2"
source /opt/ros/jazzy/setup.bash
source /opt/ros2_ws/install/setup.bash
command -v ros2 >/dev/null || fail "ros2 missing"
ros2 pkg prefix px4_msgs >/dev/null || fail "px4_msgs missing"

echo "[3/8] Gazebo"
command -v gz >/dev/null || fail "gz missing"
gz sim --version

echo "[4/8] Micro XRCE-DDS Agent"
MicroXRCEAgent udp4 -p 8888 \
    >"${TEST_ROOT}/microxrce-agent.log" 2>&1 &
AGENT_PID=$!
sleep 2
kill -0 "${AGENT_PID}" 2>/dev/null || fail "MicroXRCEAgent failed to start"

echo "[5/8] Gazebo -> ROS /clock bridge"
ros2 run ros_gz_bridge parameter_bridge \
    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" \
    >"${TEST_ROOT}/clock-bridge.log" 2>&1 &
BRIDGE_PID=$!
sleep 2
kill -0 "${BRIDGE_PID}" 2>/dev/null || fail "clock bridge failed to start"

echo "[6/8] PX4 SITL + Gazebo"
cd /opt/PX4-Autopilot
HEADLESS=1 make px4_sitl gz_x500 \
    >"${TEST_ROOT}/px4-gazebo.log" 2>&1 &
PX4_PID=$!

echo "[7/8] Waiting for PX4 ROS 2 data"
DEADLINE=$((SECONDS + TIMEOUT_SEC))
while (( SECONDS < DEADLINE )); do
    if ros2 topic list 2>/dev/null | grep -qx "/fmu/out/vehicle_status_v1"; then
        break
    fi
    sleep 2
done

ros2 topic list 2>/dev/null | grep -qx "/fmu/out/vehicle_status_v1" || {
    echo "---- PX4 log ----" >&2
    tail -n 120 "${TEST_ROOT}/px4-gazebo.log" >&2 || true
    echo "---- XRCE log ----" >&2
    tail -n 120 "${TEST_ROOT}/microxrce-agent.log" >&2 || true
    fail "PX4 vehicle_status did not appear within ${TIMEOUT_SEC}s"
}

ros2 topic echo --once \
    --qos-reliability best_effort \
    /fmu/out/vehicle_status_v1 \
    >"${TEST_ROOT}/vehicle-status.txt" 2>&1 \
    || fail "vehicle_status could not be read"

echo "[8/8] Time synchronization + simulation clock"

ros2 topic echo --once \
    --qos-reliability best_effort \
    /fmu/out/timesync_status \
    >"${TEST_ROOT}/timesync-status.txt" 2>&1 \
    || fail "timesync_status could not be read"

ros2 topic echo --once \
    /clock \
    >"${TEST_ROOT}/clock.txt" 2>&1 \
    || fail "Gazebo /clock did not reach ROS 2"

echo
echo "===== RUNTIME INTEGRATION TEST PASSED ====="
echo "Artifacts: ${TEST_ROOT}"
EOF

RUN chmod +x /usr/local/bin/dronelab-integration-test

# ============================================================
# 21. ENTRYPOINT
# ============================================================

RUN cat > /usr/local/bin/dronelab-entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash

if [[ -f /opt/ros2_ws/install/setup.bash ]]; then
    source /opt/ros2_ws/install/setup.bash
fi

if [[ -f /opt/venv/bin/activate ]]; then
    source /opt/venv/bin/activate
fi

export TZ="${TZ:-Etc/UTC}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

exec "$@"
EOF

RUN chmod +x /usr/local/bin/dronelab-entrypoint.sh

# ============================================================
# 22. FINAL USER
# ============================================================

USER ${USERNAME}
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/dronelab-entrypoint.sh"]
CMD ["bash"]
