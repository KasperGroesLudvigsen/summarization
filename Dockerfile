FROM nvcr.io/nvidia/pytorch:23.12-py3


RUN git clone https://github.com/huggingface/alignment-handbook.git
RUN cd ./alignment-handbook/
RUN python -m pip install .

RUN python -m pip install flash-attn --no-build-isolation