import os
import subprocess

model_dir = "app/models/"
version = "1.0.0"

os.environ["model_dir"] = os.path.abspath(model_dir)

command = "nohup tensorflow_model_server --rest_api_port=8501 --model_name=cifar10 --model_base_path=\"${model_dir}\" > server.log 2>&1"

processo = subprocess.Popen(
    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
saida, erros = processo.communicate()


