FROM horovod/horovod:0.25.0

# These args are overriden by run.sh to match the current user, ids and home dir.

ARG REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/mf.crt
ARG CURL_CA_BUNDLE=/usr/local/share/ca-certificates/mf.crt
COPY mf.crt /usr/local/share/ca-certificates/mf.crt

ENV TORCH_HOST=download.pytorch.org/whl/cu117
ENV MY_APT='apt -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

RUN update-ca-certificates

RUN $MY_APT update && $MY_APT install -y software-properties-common cuda-compat-11-7 sudo && \
add-apt-repository ppa:ubuntugis/ppa && $MY_APT update && \
$MY_APT install -y gdal-bin libgeos-dev git vim nano sudo libx11-dev \
tk python3-tk tk-dev libpng-dev libffi-dev dvipng ninja-build \
texlive-latex-base openssh-server netcat libeccodes-dev libeccodes-tools && \
apt-get clean all

RUN pip install --extra-index-url https://$TORCH_HOST --trusted-host $TORCH_HOST --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org torch==2.0.0+cu117 torchvision

RUN pip install --force-reinstall --extra-index-url https://$TORCH_HOST --trusted-host $TORCH_HOST --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org horovod==0.27.0

COPY requirements.txt /root/requirements.txt
RUN set -eux && pip install --upgrade pip && pip install -r /root/requirements.txt


