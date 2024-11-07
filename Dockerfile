## Build Stage
#FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04 AS builder
#
#USER root
##RUN USER=root cargo new deciduously-com
#
##RUN apt-get -y update --allow-insecure-repositories && apt-get install -y curl libssl-dev pkg-config
##RUN apt-get update --allow-insecure-repositories
#RUN apt-get update && apt-get install curl
##RUN apt-get install -y curl libssl-dev pkg-config
#
#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
#
#RUN . "$HOME/.cargo/env"
#
#WORKDIR /opt/code
#COPY Cargo.toml Cargo.lock ./
#COPY src ./src
#
#RUN cargo build --features cuda --release

# Bundle Stage
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
COPY synap-forge-llm .

#RUN nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version,compute_cap --format=csv

USER 1000
EXPOSE 8000
ENTRYPOINT ["./synap-forge-llm"]