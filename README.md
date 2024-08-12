# Help needed🥲
Please help me to understand how to fix the issue with  
```
ModuleNotFoundError: No module named 'src'
```

# This repo 
The goal of this repo is to share local implementation for the [Using-Torch-FedAvg-on-MNIST-dataset tutorial](https://docs.substra.org/en/stable/examples/substrafl/get_started/run_mnist_torch.html#List-results)

The tutorial was implemented up to the [List results section (including the "List results")](https://docs.substra.org/en/stable/examples/substrafl/get_started/run_mnist_torch.html#List-results)


# Retro steps 
1) CD repo 
```
cd C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset #swap with your repo path 
```

2) Create a new env 

```
conda env create -f substra-environment-torch_fedavg_assets.yml
```
3) Activate the env 
```
conda activate torch_fedavg_assets

```

4) Run the app 
```
python -m src.main
```

5) Get the error: 


```
(torch_fedavg_assets) C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset>python -m src.main
Number of organizations: 3
Algorithm provider: MyOrg1MSP
Data providers: ['MyOrg2MSP', 'MyOrg3MSP']
MNIST data prepared successfully.
Data registered successfully.
Dataset keys: {'MyOrg2MSP': '8225bd51-7fd4-4fae-a3d9-ac5c5f9ef075', 'MyOrg3MSP': 'fa86b75e-e0a6-43d3-b0cf-0f3000517b2b'}
Train datasample keys: {'MyOrg2MSP': '70373be7-6870-4674-978a-871ff315660f', 'MyOrg3MSP': '4e46c822-9b92-45ab-a3e4-a5ecc785cb8a'}
Test datasample keys: {'MyOrg2MSP': 'b2617eca-a93e-401d-a4b7-3f731bfb26a9', 'MyOrg3MSP': 'f9dadf77-e0d2-4cbe-9301-5c531ab817a3'}
Strategy, nodes, and evaluation components created successfully.
Rounds progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 3016.76it/s]
Compute plan progress:   0%|                                                                                                                                                                          | 0/21 [00:00<?, ?it/s]Traceback (most recent call last):
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\local-worker\tmptznku85x\function.py", line 13, in <module>
    remote_struct = RemoteStruct.load(src=Path(__file__).parent / 'substrafl_internal')
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substrafl\remote\remote_struct.py", line 94, in load
    instance = cloudpickle.load(f)
               ^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'src'
Compute plan progress:   0%|                                                                                                                                                                          | 0/21 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\compute\spawner\subprocess.py", line 114, in spawn
    subprocess.run(py_command, capture_output=False, check=True, cwd=function_dir, env=envs)
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['C:\\Users\\probl\\anaconda3\\envs\\torch_fedavg_assets\\python.exe', 'C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\function.py', '@C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\tmpgcef6o7l\\arguments.txt']' returned non-zero exit status 1.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\src\main.py", line 59, in <module>
    main()
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\src\main.py", line 45, in main
    compute_plan = run_experiment(
                   ^^^^^^^^^^^^^^^
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\src\experiment_config.py", line 32, in run_experiment
    return execute_experiment(
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substrafl\experiment.py", line 498, in execute_experiment
    compute_plan = client.add_compute_plan(
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\client.py", line 48, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\client.py", line 548, in add_compute_plan
    return self._backend.add(spec, spec_options=spec_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 487, in add
    compute_plan = add_asset(spec, spec_options)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 406, in _add_compute_plan
    compute_plan = self.__execute_compute_plan(spec, compute_plan, visited, tasks, spec_options)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 269, in __execute_compute_plan
    self.add(
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 491, in add
    add_asset(key, spec, spec_options)
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 437, in _add_task
    self._worker.schedule_task(task)
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\compute\worker.py", line 359, in schedule_task
    self._spawner.spawn(
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\compute\spawner\subprocess.py", line 116, in spawn
    raise ExecutionError(e)
substra.sdk.backends.local.compute.spawner.base.ExecutionError: Command '['C:\\Users\\probl\\anaconda3\\envs\\torch_fedavg_assets\\python.exe', 'C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\function.py', '@C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\tmpgcef6o7l\\arguments.txt']' returned non-zero exit status 1.

(torch_fedavg_assets) C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset>
```

# Used local machine specification 
System Information: OS: Windows 11 10.0.22631 Machine: AMD64

CPU Information: AMD Ryzen 7 7840HS w/ Radeon 780M Graphics 8 cores, 16 threads

Memory Information: Total: 31.22GB

Disk Information: disk_C:: Total: 930.43GB

GPU Information: NVIDIA GeForce RTX 4070 Laptop GPU: Total memory: 8188.0MB