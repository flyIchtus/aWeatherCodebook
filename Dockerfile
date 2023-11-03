FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GUID
ARG HOME_DIR
ARG INJECT_MF_CERT
ARG REQUESTS_CA_BUNDLE
ARG CURL_CA_BUNDLE
ARG NODE_EXTRA_CA_CERTS

ARG REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/mf.crt
ARG CURL_CA_BUNDLE=/usr/local/share/ca-certificates/mf.crt
COPY mf.crt /usr/local/share/ca-certificates/mf.crt


ENV MY_APT='apt -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

RUN update-ca-certificates

RUN $MY_APT update && $MY_APT install -y ninja-build build-essential


COPY requirements.txt /root/requirements.txt
RUN set -eux && pip install --upgrade pip && pip install -r /root/requirements.txt

RUN set -eux && groupadd --gid $USER_GUID $GROUPNAME \
    # https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
    && mkdir -p $HOME_DIR && useradd -l --uid $USER_UID --gid $USER_GUID -s /bin/bash --home-dir $HOME_DIR --create-home $USERNAME \
    && chown $USERNAME:$GROUPNAME $HOME_DIR \
    && echo "$USERNAME:$USERNAME" | chpasswd




