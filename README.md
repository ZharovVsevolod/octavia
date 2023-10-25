# Project Octavia  
Как осуществлялся прогон обучения на google colab:  
[ссылка на wandb](https://wandb.ai/wsewolod/project_octavia/runs/irzh7hm0/workspace?workspace=user-wsewolod)

## Проблема с linux
Если запускать проект на linux-системе, то, поскольку poetry плохо работает с pytorch+cuda, в pyproject.toml надо изменить ссылки импорта на следующие:  
torch = { url = "https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl"}  
torchaudio = { url = "https://download.pytorch.org/whl/cu118/torchaudio-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl"}  
torchvision = { url = "https://download.pytorch.org/whl/cu118/torchvision-0.16.0%2Bcu118-cp310-cp310-linux_x86_64.whl"}  
torchtext = { url = "https://download.pytorch.org/whl/torchtext-0.16.0%2Bcpu-cp310-cp310-linux_x86_64.whl"}  
