# BixBench-env for linux/amd64.
#
# BixBench's harness runs its agent in futurehouse/bixbench:aviary-notebook-env
# (sha256:5f5dea50a0d37d252bcb1271c7fb843c0ce2e5fda4961279ca27eccd01208b5e), published for
# linux/arm64 only. Its recipe is packages/notebook/docker/Dockerfile.pinned of
# Future-House/aviary at dc6fe209cddb7b99343ff26bae3d4ac607ea0b19: every package pinned there
# matches the image's layer history. That recipe leaves transitive packages, and conda itself,
# unpinned, so building it today resolves different versions.
#
# This file follows the recipe's steps, but its conda environment is the published image's:
# conda-pins.txt lists every conda package of that image at its version, and
# conda-linux-64.lock is those pins solved for linux-64, which is installed here as /opt/conda
# without a solve. requirements.txt is the recipe's, kept for reference.
#
# Nothing is taken from Anaconda's repositories, whose terms restrict use by large
# organizations: the recipe's continuumio/miniconda3:24.9.2-0 base is replaced by the Debian
# release it is built on plus the same apt packages and shell setup, the lock uses
# conda-forge builds where the published image has Anaconda's, and conda is configured for
# conda-forge and bioconda.
#
#   docker build --platform linux/amd64 -t ghcr.io/allenai/olmo-eval-bixbench-env:<tag> \
#     src/olmo_eval/evals/external/benchmarks/bixbench/sandbox
FROM debian:12.7-slim@sha256:36e591f228bb9b99348f584e83f16e012c33ba5cad44ef5981a1d7c0a93eca22

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# What continuumio/miniconda3 installs around conda.
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 ca-certificates git libglib2.0-0 libsm6 libxext6 libxrender1 mercurial \
        openssh-client procps subversion wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=mambaorg/micromamba:2.0.5@sha256:8d0d4cad074afe3ac93b58dce88bd2659153d3fa6de3a6968f501710e8d3c832 \
    /bin/micromamba /usr/local/bin/micromamba

RUN mkdir /workspace && \
    mkdir /envs
WORKDIR /envs

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        wget \
        gpg \
        software-properties-common \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY conda-pins.txt conda-linux-64.lock requirements.txt ./
RUN micromamba create -y -p /opt/conda --file conda-linux-64.lock && \
    micromamba clean --all -y
# micromamba skips post-link scripts, which conda runs. Bioconda's annotation data packages
# (org.Hs.eg.db, GO.db, ...) install their data into R from them, checking each download's MD5.
RUN for script in /opt/conda/bin/.*-post-link.sh; do \
        PREFIX=/opt/conda bash "$script" || exit 1; \
    done

RUN printf 'channels:\n  - conda-forge\n  - bioconda\n' > /opt/conda/.condarc && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN python -m ipykernel install --user --name python3 --display-name "Python 3 (ipykernel)"
RUN R -e 'IRkernel::installspec(name = "R", displayname = "R (4.3.3)")'

# Certain tools are not easily installable via conda. A common practice for
# bioinformaticians is to use udocker to run certain heavy duty omics processing
# tools in an isolated environment
RUN udocker --allow-root install && \
    udocker --allow-root pull ezlabgva/busco:v5.8.0_cv1

WORKDIR /workspace

RUN conda clean --all -f -y && \
    rm -rf /root/.cache/pip
