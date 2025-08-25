FROM subtle/base_cu12_ubuntu20_py310:latest

# Define project directory variable
ARG APP_DIR=SubtleCT_DNE_Infer_v2
ENV APP_PATH=/root/${APP_DIR}

# Set working directory (optional)
WORKDIR /root

# Copy code into the image
COPY ${APP_DIR} ${APP_PATH}

# Create symbolic links for python3.10
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3

# Install dependencies (from Huawei Cloud mirror)
RUN pip install -i https://repo.huaweicloud.com/repository/pypi/simple \
    torch==2.3.1 \
    SimpleITK==2.3.1 \
    pydicom==2.4.4 \
    tqdm \
    PyYAML \
    numpy

# Default entrypoint: bash
CMD ["bash"]