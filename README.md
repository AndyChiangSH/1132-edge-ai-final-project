# 1132-edge-ai-final-project

**Group 37: 江尚軒、蔡昀叡、陳建樺、李任本耀**

## Our result

Our best result is in [`result/result_0529_lora_512.csv`](https://github.com/AndyChiangSH/1132-edge-ai-final-project/blob/main/result/result_0529_lora_512.csv)

## Instructions to reproduce

1. Clone this GitHub repo
    
    ```bash
    git clone https://github.com/AndyChiangSH/1132-edge-ai-final-project.git
    ```
    
2. Move into this repo
    
    ```bash
    cd 1132-edge-ai-final-project
    ```
    
3. Create and activate the venv
    
    ```bash
    python3 -m venv 1132-edge-ai-final-project
    source 1132-edge-ai-final-project/bin/activate
    ```
    
4. Install the required packages from pip
    
    ```bash
    pip install -r requirements.txt
    pip install gemlite==0.4.4
    ```
    
5. Run this script to generate the result
    
    ```bash
    sh run.sh
    ```
    
6. The result will be saved in `result/result.csv`