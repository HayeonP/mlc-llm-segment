- Base commit: `72a184f2047272ff482c352ee0ea2191861f987f`
- Instal tvm-segment for TVM v0.21.0
- Create conda env
    ```bash
    conda create -n mlc-llm -c conda-forge -c huggingface \
        "python=3.10" \
        "cmake=3.26" \
        "numpy=1.26.4" \
        "pillow>=5.3.0" \
        psutil \
        typing_extensions \
        pydantic \
        shortuuid \
        fastapi \
        requests \
        tqdm \
        prompt_toolkit \
        ml_dtypes \
        rust \
        transformers
    ```
- Install few more packages
    ```bash
    # pytorch
    python3 -m pip install https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl

    # accelerate
    python3 -m pip install acclerate
    ```

- Build mlc-llm
    ```bash
    # clone from GitHub
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/
    # create build directory
    mkdir -p build && cd build
    # generate build configuration
    python ../cmake/gen_cmake_config.py
    # build mlc_llm libraries
    cmake .. && cmake --build . --parallel $(nproc) && cd ..

    #(Optional) To enable debug mode,
    cmake -DENABLE_DEBUG_MESSAGES=ON ..
    cmake --build . --parallel $(nproc) && cd ..
    ```

- Download Llama3.2-1B-Instruct
    - Get authentification from the llama huggingface repository (`https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`)

    - Clone the llama projecct
    ```bash
    # Login to huggingface
    huggingface-login

    # In a model diretory,
    cd dist
    git lfs install
    git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
    ```

- Compilation (With quantization)

    (1) Create config
    ``` bash
    # NOTE: When the model direcotry is ~/workspace/llm_models/Llama-3.2-1B-Instruct
    mlc_llm gen_config ~/workspace/llm_models/Llama-3.2-1B-Instruct/ \
        --quantization q4f16_1 \
        --conv-template phi-2 \
        -o dist/Llama-3.2-1B-Instruct-q4f16_1-MLC
    ```

    (2) Convert weight
    ```bash
    mlc_llm convert_weight ~/workspace/llm_models/Llama-3.2-1B-Instruct/ \
        --quantization q4f16_1 \
        -o dist/Llama-3.2-1B-Instruct-q4f16_1-MLC
    ```

    (3) Compile
    ```bash
    mlc_llm compile dist/Llama-3.2-1B-Instruct-q4f16_1-MLC --output dist/Llama-3.2-1B-Instruct-q4f16_1-MLC/Llama-3.2-1B-Instruct-q4f16_1-MLC.so
    ```