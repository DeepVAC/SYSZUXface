FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
LABEL maintainer "Gemfield <gemfield@civilnet.cn>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim locales libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple numpy opencv_python

ENV GEMFIELD=/opt/gemfield/
RUN mkdir -p ${GEMFIELD}
COPY src ${GEMFIELD}/

WORKDIR ${GEMFIELD}/
ENTRYPOINT [ "python", "product_line.py" ]


