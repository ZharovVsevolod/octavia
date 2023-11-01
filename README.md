# Project Octavia  
[ссылка на wandb](https://wandb.ai/lost_in_thoughts/project_octavia)

## Проблема с linux
Если запускать проект на linux-системе, то в pyproject.toml надо изменить ссылки импорта на следующие:  
torch = {url = "https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl"}  
