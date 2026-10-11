FROM runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204
LABEL org.opencontainers.image.description="rAidiology CT anatomy names, research use, not a medical device"
RUN python -m pip install --no-cache-dir runpod==1.12.0 nibabel==5.4.2 pandas==2.2.3 scipy==1.15.3 psutil==7.2.2
RUN python -c "import runpod, nibabel, pandas, scipy, psutil, torch"
COPY . /opt/raid-serve
ENV RAID_SERVE_HOME=/opt/raid-serve \
    RAID_SERVE_REPO=/opt/raid-serve/repo \
    RAIDIOLOGY_MODELS=/opt/raid-serve/Models \
    RAID_CT_VERSION=2 \
    PYTHONUNBUFFERED=1
CMD ["python", "-u", "/opt/raid-serve/handler.py"]
