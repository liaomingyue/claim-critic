# verl 官方镜像只带 torch / vLLM 等框架，不带 verl 本身；按官方说明用 --no-deps 安装
FROM verlai/verl:vllm024.dev2

RUN pip3 install --no-cache-dir --no-deps verl==0.9.1 \
 && pip3 install --no-cache-dir "pyyaml>=6" "openai>=1.40" pytest
