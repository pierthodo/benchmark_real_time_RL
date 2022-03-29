ARG BASE
FROM ${BASE}


RUN git clone https://github.com/pierthodo/rtrl.git \
  && cd /app/rtrl \
  && pip --no-cache-dir install -e .


RUN git clone https://github.com/pierthodo/mbrl-lib.git \
  && cd /app/mbrl-lib \
  && pip install -e ".[dev]"
