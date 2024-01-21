## AI-Lab-Machine-Guide

Welcome to the GPU GitHub repository guide! This guide provides step-by-step instructions for setting up and utilizing GPU-based AI and machine learning systems. Below is an outline of the topics covered in this guide.

### Table of Contents

- [Basic Linux Usage](#basic-linux-usage)
   - [Connecting to a Remote Linux Machine](#connecting-to-a-remote-linux-machine)
- [What is Conda?](#what-is-conda)
    - [Installing Conda](#installing-conda)
    - [Setting up Conda Environments](#setting-up-conda-environments)
- [Installing PyTorch and TensorFlow](#installing-pytorch-and-tensorflow)
    - [PyTorch](#pytorch)
    - [TensorFlow](#tensorflow)
- [What is Screen?](#what-is-screen)
    - [Setting up Screen](#setting-up-screen)
    - [How to Use Screen](#how-to-use-screen)
- [Special Topics](#special-topics)
    - [Handling Multi-GPU Systems](#handling-multi-gpu-systems)
    - [Opening a Browser on the Machine](#opening-a-browser-on-the-machine)
    - [Using Jupyter Notebook](#using-jupyter-notebook)
- [Quick Reference](#quick-reference)

### Basic Linux Usage

To get started with using Linux, we highly recommend watching the [Basic How to Use Linux video](https://www.youtube.com/watch?v=gd7BXuUQ91w) that provides a comprehensive overview of Linux fundamentals.

### Connecting to a Remote Linux Machine

Make sure to connect to the [AUS VPN](https://servicenow.aus.edu/sp?sys_kb_id=d71b18c1872fb150f9c863de8bbb354b&id=kb_article_view&sysparm_rank=1&sysparm_tsqueryId=c92777e293377110d761b4908bba10ff) before connecting to the Remote Linux Machine.

#### Using Terminal

The built-in Terminal application provides a straightforward way to connect to a remote Linux machine using SSH.

1. Open Terminal and use the `ssh` command to establish an SSH connection to the remote Linux machine:

   ```shell
   ssh username@remote-linux-machine
   ```

   Replace `username` with your actual username and `remote-linux-machine` with the IP address or hostname of the remote Linux machine.

#### Using MobaXterm for Windows

MobaXterm is a powerful terminal software for Windows that integrates various network tools, including an SSH client, into a single application. Here's how you can use MobaXterm to connect to a remote Linux machine:

1. Download and install MobaXterm on your Windows machine.

2. Launch MobaXterm and click on the "Session" button in the toolbar.

3. In the "Session Settings" window, select "SSH" as the session type.

4. Enter the IP address or hostname of the remote Linux machine in the "Remote host" field.

5. Provide your username in the "Specified username" field.

6. Click "OK" to initiate the SSH connection to the remote Linux machine.

MobaXterm provides a user-friendly interface with tabbed terminal windows, easy file transfer functionality, and advanced SSH capabilities, making it a convenient tool for remotely accessing and managing Linux systems from a Windows environment.

### What is Conda?

Conda is an open-source package management system and environment management system designed to simplify the installation and management of software packages.

#### Installing Conda

To install Conda on your GPU machine, follow these steps:

1. Open a terminal and use the following command to download the installer:

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
```

You can find all the available versions of the installer [here](https://repo.anaconda.com/archive/).

2. Run the installer by executing the following command:

```shell
bash Anaconda3-2023.07-2-Linux-x86_64.sh
```

3. Press Enter to review the license agreement. Then press and hold Enter to scroll through the license text.

4. Enter `yes` to agree to the license agreement.

5. The installer allows you to choose the installation location. To accept the default install location, press Enter. If you want to specify an alternate installation directory, enter the desired path. Note that the default path is shown as `PREFIX=/home/<USER>/anaconda3`. The installation process may take a few minutes to complete.

6. After the installation, Anaconda recommends initializing Anaconda Distribution by running `conda init`. If you want to skip initialization, you can enter `no`. However, it's recommended to enter `yes` to initialize Conda.

   If you choose not to initialize Conda, you can manually initialize it after the installation using the following commands (replace `<PATH_TO_CONDA>` with the path to your Conda install):

   ```shell
   source <PATH_TO_CONDA>/bin/activate
   conda init
   ```

7. Once the installation is complete, the installer will display "Thank you for installing Anaconda3!".

8. Close and re-open your terminal window for the installation to take effect. Alternatively, you can run the following command to refresh the terminal:

```shell
source ~/.bashrc
```

9. Optionally, you can configure whether or not the base environment is activated by default when opening a new shell. Use one of the following commands:

- To activate the base environment by default:

```shell
conda config --set auto_activate_base True
```

- To deactivate the base environment by default:

```shell
conda config --set auto_activate_base False
```

Note: The above commands only work if `conda init` has been run first. `conda init` is available in Conda versions 4.6.12 and later.

If you find any difficulty in installing conda you can refer to this [video](https://www.youtube.com/watch?v=P6eGTN9QN2Q), they follow similar steps.

Congratulations! You have successfully installed and set up Conda on your GPU machine.

### What are Environments?

Environments in Conda allow you to create isolated and self-contained spaces where you can install specific versions of packages and dependencies that are required for a particular project. Each environment can have its own set of packages, Python version, and dependencies without interfering with other environments or the base environment.

#### When to Create Environments

Creating environments is particularly useful in the following scenarios:

1. **Project Isolation:** If you are working on multiple projects that require different versions of packages or dependencies, you can create separate environments for each project. This ensures that changes made in one environment do not affect the packages or dependencies used in another project.

2. **Dependency Management:** Environments help manage complex dependencies between packages and ensure that all required packages are installed with their compatible versions. This is especially crucial when working with complex machine learning or AI projects that have multiple dependencies.

3. **Reproducibility:** By creating an environment for your project, you can easily share the environment configuration with others. This enables collaborators to recreate the exact same environment, ensuring reproducibility of your work.

#### Importance of Avoiding the Base Environment

While the base environment in Conda is pre-installed and serves as a default environment, it is generally recommended not to directly install packages or dependencies into the base environment. Here's why:

1. **Maintain Clean State:** By not cluttering the base environment with project-specific packages, you maintain a clean and basic setup that can be used as a starting point for creating new environments.

2. **Avoid Version Conflicts:** Installing packages directly into the base environment increases the chances of version conflicts, as different projects may require different versions of the same package. By creating separate environments, you can ensure that each project has its own isolated set of packages, preventing version conflicts.

3. **Simplify Dependency Tracking:** Environments allow you to keep track of the exact package versions used in different projects. This simplifies troubleshooting and ensures that the same versions are used when recreating the environment.

By creating separate environments, you can effectively manage project-specific dependencies, version requirements, and ensure reproducibility while keeping the base environment clean and minimal.

### Setting up Conda Environments

To set up Conda environments and manage them effectively, follow these guidelines:

#### Creating an Environment

1. To create a new environment, open a terminal and execute the following command:

   ```shell
   conda create --name myenv
   ```

   Replace `myenv` with your desired environment name. Conda will create a new environment with the specified name.

2. You can also choose a specific Python version for your environment by appending the desired version number at the end of the command. For example, to create an environment with Python 3.8, use:

   ```shell
   conda create --name myenv python=3.8
   ```

   This will create an environment named `myenv` with Python 3.8 as the default Python version.

#### Activating an Environment

1. To activate an environment, use the following command:

   ```shell
   conda activate myenv
   ```

   Replace `myenv` with the name of the environment you want to activate.

   Note: The environment name will be displayed in the terminal prompt, indicating that the environment is active.

#### Deactivating an Environment

1. To deactivate the current environment and return to the base environment, use the following command:

   ```shell
   conda deactivate
   ```

   This will deactivate the currently active environment and revert back to the base environment.

   Note: The terminal prompt will no longer display the environment name, indicating that you are in the base environment.

#### Checking Available Environments

1. To list all available environments, execute the following command:

   ```shell
   conda info --envs
   ```

   This will display a list of all existing Conda environments on your system.

### Installing PyTorch and TensorFlow

#### PyTorch

To install PyTorch with different CUDA versions, follow the instructions below:

##### CUDA 11.4

For CUDA 11.4, you can treat it like CUDA 11.3. Run the following command to install PyTorch:

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

##### CUDA 12.2

To install PyTorch with CUDA 12.2, use the following command:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Verifying the Install

To verify the PyTorch installation, run the following command in your activated environment:

```shell
python3 -c "import torch; print(torch.cuda.is_available())"
```

If the installation is successful, it will print `True` if PyTorch can detect the CUDA capabilities of your GPU.

#### TensorFlow

To install TensorFlow with CUDA versions greater than or equal to 12.0, follow these steps:

1. Create a new environment for TensorFlow installation:

   ```shell
   conda create --name tf python=3.10
   ```

2. Activate the newly created environment:

   ```shell
   conda activate tf
   ```

3. Install the required CUDA and cuDNN dependencies:

   ```shell
   conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
   ```

4. Set the environment variable for the library path:

   ```shell
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   ```

5. Sign out and sign back in via SSH or close and reopen your terminal window to activate the changes.

6. Reactivate your Conda session and install TensorFlow:

   ```shell
   python3 -m pip install tensorflow==2.10
   ```

#### Verifying the Install

To verify the TensorFlow installation and check the number of available GPUs, run the following command:

```shell
python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
```

If the installation is successful, it will print the number of available GPUs.

Congratulations! You have successfully installed PyTorch and TensorFlow with the specified CUDA versions. You can now proceed with your GPU-based AI and machine learning projects.

### What is Screen?

In certain instances, you may need to run scripts or processes that require a significant amount of time to complete. However, keeping your local machine or laptop powered on for extended periods of time is not practical or feasible. This is where the utility of Screen comes into play.

Screen is a terminal multiplexer that allows you to create and manage multiple terminal sessions within a single window. It enables you to run long-running processes, such as scripts or programs, while detaching from the session and even logging out or disconnecting from the remote server. The processes running within the Screen session continue to execute in the background, even if your SSH or terminal connection is interrupted.

#### Setting up Screen

To set up Screen on your machine, follow these steps:

1. Open a terminal window on your remote server or local machine.
2. Install Screen by running the following command:

```shell
conda install conda-forge::screen
```

3. Wait for the installation to complete.

#### How to Use Screen

To use Screen effectively, follow these commands:

1. Start a new Screen session by running the command:

```shell
screen -r screenName
```

Replace `screenName` with your desired screen name. Screen will create a new session with the specified name.

2. Within the Screen session, you can run your script or start any process that you want to run in the background for a long duration.

3. To detach from the Screen session and leave the process running in the background, press `Ctrl+A`, followed by the `D` key. This command tells Screen to detach from the session.

4. You can now safely log out or disconnect from the remote server without interrupting the processes running in the Screen session.

#### Reattaching to a Screen Session

To reattach to a previously detached Screen session and resume working with the processes running in the background, follow these steps:

1. Open a terminal and connect to your remote server or open a new terminal window on your local machine.

2. Run the following command to list all available Screen sessions:

```shell
screen -ls
```

3. Identify the detached Screen session you want to reattach to, and note its session name.

4. To reattach to the Screen session, use the following command:

```shell
screen -r <session_name>
```

Replace `<session_name>` with the name of the Screen session you want to reattach to.

5. You will be reconnected to the Screen session, and you can interact with the processes running within it as if you never detached from the session.

### Special Topics

This section covers advanced topics for specific scenarios.

### Handling Multi-GPU Systems

When working with a multi-GPU system, it is essential to properly configure your environment to utilize the desired GPUs effectively. Here are some guidelines to help you handle multi-GPU systems:

#### Setting the Visible CUDA Devices

Before running any script, you can specify which GPUs should be used by setting the `CUDA_VISIBLE_DEVICES` environment variable. This variable allows you to select specific GPUs for your script to utilize.

To define the visible CUDA devices, you have two options:

1. **Command Line Method:**

   Use the following command before executing your script:

   ```shell
   CUDA_VISIBLE_DEVICES=0,1 python script.py
   ```

   Replace `0,1` with the indexes of the desired GPUs you want to use. This will restrict the script to only use the specified GPUs in your multi-GPU system.

2. **Code Method:**

   Alternatively, you can include the following code snippet at the beginning of your script:

   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
   ```

   This will ensure that the script only uses the GPUs specified in the `CUDA_VISIBLE_DEVICES` environment variable.

Properly configuring the visible CUDA devices ensures that your scripts and processes run on the desired GPUs and take full advantage of the available computing power.

### Opening a Browser on the Machine

In some cases, you may need to open a browser directly on your GPU machine. Here's how you can achieve this:

1. Activate the desired Conda environment:

```shell
conda activate myenv
```

Replace `myenv` with the name of the environment you want to activate.

2. Set the environment variables for the display and Xauthority:

```shell
export DISPLAY=localhost:10.0
export XAUTHORITY=$HOME/.Xauthority
```

This ensures that the browser knows where to display its graphical interface.

3. Open the browser using the following command:

```shell
firefox https://google.com
```

Replace `https://google.com` with the desired URL or web page you want to open in the browser. 

Note: If Firefox is not installed on your machine, you can install it using the appropriate package manager for your operating system.

By following these steps, you will be able to launch a browser on your GPU machine directly from the terminal, allowing you to perform any necessary web-related tasks or access online resources.

### Using Jupyter Notebook

When working with Jupyter Notebook on a remote server, it is essential to establish a secure connection and access the Jupyter environment through a web browser. Here's how you can accomplish this:

1. Connect to the AUS VPN to securely access the remote server.

2. Open the command prompt (CMD) on your local laptop and run the following command to establish an SSH tunnel:

   ```shell
   ssh -L 8888:localhost:8888 username@10.00.00.000
   ```

   Replace `username` with your specific username and `10.00.00.000` with the appropriate IP address of the remote server.

3. Enter the password when prompted to authenticate the SSH connection.

4. After successful authentication, you will receive a URL in the command prompt. Copy and paste this URL into your web browser to access Jupyter Notebook on the remote server.

   For example:

   ```plaintext
   http://localhost:8888/notebooks/
   ```

   This will allow you to work with Jupyter Notebook as if it were running on your local machine, providing a seamless and efficient workflow for data analysis and code development.

### Quick Reference

This section provides a quick reference for useful commands.

- `du -sh`: Displays the total disk usage of a directory.
- `ls -1 | wc -l`: Counts the number of files and directories in the current directory.
- `nvidia-smi`: Shows GPU usage and information for NVIDIA GPUs.
- `top`: Displays real-time system monitoring information, including CPU usage, memory usage, and running processes.
- `htop`: An interactive version of `top` that provides a more user-friendly and detailed view of system resources.
- `free -h`: Displays the memory usage and availability.
- `grep`: Searches for a specific pattern in a file or command output.
- `find`: Searches for files or directories based on criteria such as name, size, or permissions.
- `tar`: Creates or extracts compressed archives.
- `pip`: Package installer for Python packages.
- `conda`: Package and environment management system for Python and other programming languages.
- `jupyter notebook`: Launches the Jupyter Notebook application for interactive coding and data analysis.
- `ssh`: Establishes a secure shell connection to a remote server.
- `scp`: Securely copies files between local and remote systems.
- `curl`: Command-line tool for making requests to URLs or APIs.
- `wget`: Command-line tool for downloading files from the web.

<div align="center">
  
## Happy Coding! 💻🚀

Congratulations! You have successfully explored this guide, equipping yourself with the knowledge to set up and harness the power of GPU-based AI and machine learning systems. 

This comprehensive overview empowers you to embark on exciting AI and data science endeavors. Remember, this is just the beginning of your journey into the world of cutting-edge technology!

Feel free to delve deeper into the provided links and references to expand your understanding and delve into more advanced topics. Let your curiosity guide you!

Remember, every line of code you write, every algorithm you create, and every model you train has the potential to transform industries, solve complex problems, and shape the future.

Now, go forth and let your creativity soar! Happy coding! 🚀

</div>