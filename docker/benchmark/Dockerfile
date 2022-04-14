ARG BASE
FROM ${BASE}


RUN git clone https://github.com/pierthodo/rtrl.git \
  && cd /app/rtrl \
  && pip --no-cache-dir install -e .


RUN git clone https://github.com/pierthodo/mbrl-lib.git \
  && cd /app/mbrl-lib \
  && pip install -e ".[dev]"

RUN git clone https://github.com/danijar/dreamerv2.git \
  && cd /app/dreamerv2 \
  && pip install -e .

RUN git clone https://github.com/pierthodo/spinningup.git \
  && cd /app/spinningup \
  && pip install -e .
# RUN pip install tensorflow==2.4.2 tensorflow_probability==0.12.2 ruamel.yaml 

RUN git clone https://github.com/pierthodo/ray.gitt \
  && cd /app/ray \
  && pip install -e .[rllib]


RUN pip install --upgrade numpy
