#第一阶段：安装基本镜像这个了
FROM continuumio/miniconda3:latest AS miniconda_stage                                           
ENV DEBIAN_FRONTEND=noninteractive                                                                                 
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt-get update 
RUN apt-get install -y patchelf
RUN apt-get install -y software-properties-common
RUN apt-get install -y unzip
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y tzdata
RUN rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"                                                                                                                                   
RUN conda install -y python=3.8                                                          
# 第二阶段：使用 CUDA 镜像                                                                     
FROM nvidia/cuda:11.6.2-devel-ubuntu18.04 AS cuda                                        
# 从 Miniconda 阶段复制环境                                                                   
COPY --from=miniconda_stage /opt/conda /opt/conda                                                
# 设置环境变量                                                                                 
ENV PATH="/opt/conda/bin:$PATH"                                                                  
# 第三阶段：构建项目                                                                           




FROM cuda   
WORKDIR /app
COPY . /app

COPY --from=0 /usr/ /docker/python
















# 安装支持
RUN conda env create -f environment.yml
RUN conda activate groundingdino
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN apt-get update
RUN apt-get  -y install ffmpeg libsm6 libxext6 
RUN pip install -v -e .


