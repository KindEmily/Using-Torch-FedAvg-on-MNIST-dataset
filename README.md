# Intro 
This repository is a private local implementation of the Substra Using Torch FedAvg on MNIST dataset tutorial 
Source: https://github.com/KindEmily/Using-Torch-FedAvg-on-MNIST-dataset 

# 🤔 Why public? 
I`d like to share the process and issues I challenged during this tutorial. 
I hope Substra developers and everyone interested in learning this tool will be able to use this knowledge and make your life a little bit easier 🥰 


# Current progress 
From tutorial https://docs.substra.org/en/stable/examples/substrafl/get_started/run_mnist_torch.html

✅ Setup
⏭️ Data and metrics (in progress) 
Please look at the separate branch to track the progress: https://github.com/KindEmily/Using-Torch-FedAvg-on-MNIST-dataset/tree/Data-and-metrics
  ✅ Data preparation
  ✅ Dataset registration
  ⏭️ Metrics definition
🚫 Machine learning components definition
  🚫 The rest of tutorial to be done 

# System info 
Note: This information represents the system state at the time of generation and 
includes only static, essential data that is unlikely to change frequently.

System Information:
  OS: Windows 11 10.0.22631
  Machine: AMD64

CPU Information:
  AMD Ryzen 7 7840HS w/ Radeon 780M Graphics
  8 cores, 16 threads

Memory Information:
  Total: 31.22GB

Disk Information:
  disk_C:\:
    Total: 930.43GB

GPU Information:
  NVIDIA GeForce RTX 4070 Laptop GPU:
    Total memory: 8188.0MB

# How to initialize a new Anaconda env 
Create the environment from the `substra-environment.yml` file:

<aside>
💡 Protip: dont forget to set the `name:` property with something meaningful
</aside>

```
cd C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset
```

```
conda env create -f substra-environment-torch_fedavg_assets.yml
```
