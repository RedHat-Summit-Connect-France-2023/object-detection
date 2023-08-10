FROM registry.redhat.io/ubi9/ubi-minimal:latest

RUN dnf install -y https://dl.min.io/client/mc/release/linux-amd64/mcli-20230808172359.0.0.x86_64.rpm && \
    dnf install -y git  && \
    dnf install -y tar && \
    dnf install -y unzip

USER 1001

ENTRYPOINT ["/bin/sh"]