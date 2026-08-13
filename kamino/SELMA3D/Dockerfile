# 1. Base image with CUDA + PyTorch + MONAI
FROM projectmonai/monai:latest

WORKDIR /app

# 2. Additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy all needed files
COPY . .

CMD ["pip", "uninstall", "torch"]
CMD ["pip", "install", "torch", "--index-url https://download.pytorch.org/whl/cu121"]
CMD ["python", "cuda_test.py"]
