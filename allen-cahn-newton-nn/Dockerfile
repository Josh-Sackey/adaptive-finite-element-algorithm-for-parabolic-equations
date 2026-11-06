# syntax=docker/dockerfile:1.7
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG DEALII_VERSION=9.7.1
ARG LIBTORCH_VERSION=2.7.1
ARG BUILD_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git ninja-build unzip \
      libboost-all-dev liblapack-dev libmetis-dev libmuparser-dev \
      libsuitesparse-dev libtbb-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Build the exact deal.II version used by the research code. Optional large
# distributed-memory packages are disabled because this solver is serial.
RUN curl -fsSL \
      "https://github.com/dealii/dealii/releases/download/v${DEALII_VERSION}/dealii-${DEALII_VERSION}.tar.gz" \
      -o /tmp/dealii.tar.gz \
    && mkdir -p /tmp/dealii \
    && tar -xzf /tmp/dealii.tar.gz --strip-components=1 -C /tmp/dealii \
    && cmake -S /tmp/dealii -B /tmp/dealii/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/dealii \
      -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
      -DDEAL_II_COMPONENT_EXAMPLES=OFF \
      -DDEAL_II_WITH_MPI=OFF \
      -DDEAL_II_WITH_P4EST=OFF \
      -DDEAL_II_WITH_PETSC=OFF \
      -DDEAL_II_WITH_SLEPC=OFF \
      -DDEAL_II_WITH_TRILINOS=OFF \
      -DDEAL_II_WITH_UMFPACK=ON \
      -DDEAL_II_WITH_LAPACK=ON \
      -DDEAL_II_WITH_TBB=ON \
    && cmake --build /tmp/dealii/build --parallel "${BUILD_JOBS}" \
    && cmake --install /tmp/dealii/build \
    && rm -rf /tmp/dealii /tmp/dealii.tar.gz

# CPU-only C++11-ABI LibTorch is portable to ordinary x86_64 compute nodes.
RUN curl -fsSL \
      "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip" \
      -o /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

WORKDIR /src
COPY CMakeLists.txt allen-cahn-newton-nn.cc neural_surrogate.cc neural_surrogate.h ./
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="/opt/dealii;/opt/libtorch" \
      -DCMAKE_INSTALL_PREFIX=/opt/allen-cahn \
    && cmake --build build --parallel "${BUILD_JOBS}" \
    && cmake --install build

FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libboost-iostreams-dev libboost-serialization-dev \
      libboost-system-dev libboost-thread-dev libgomp1 liblapack3 \
      libmetis5 libmuparser2v5 libsuitesparse-dev libtbb12 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/dealii /opt/dealii
COPY --from=builder /opt/libtorch /opt/libtorch
COPY --from=builder /opt/allen-cahn /opt/allen-cahn
COPY container/run-simulation.sh /opt/allen-cahn/bin/run-simulation
COPY container/verify-results.sh /opt/allen-cahn/bin/verify-results

RUN chmod +x /opt/allen-cahn/bin/run-simulation \
             /opt/allen-cahn/bin/verify-results \
    && mkdir -p /results

ENV LD_LIBRARY_PATH="/opt/libtorch/lib:/opt/dealii/lib:${LD_LIBRARY_PATH}"
ENV OMP_NUM_THREADS=1
WORKDIR /results
ENTRYPOINT ["/opt/allen-cahn/bin/run-simulation"]
CMD ["baseline"]

LABEL org.opencontainers.image.title="deal.II-LibTorch Allen-Cahn Example 4.7" \
      org.opencontainers.image.description="CPU reproducibility image for the quadrilateral Allen-Cahn AFEM/NN solver" \
      org.opencontainers.image.version="1.0"

