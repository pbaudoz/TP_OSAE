# TP OSAE
Hardware controller for the travaux pratiques, based on catkit2.

This repository is a public repository using catkit2 code to allow Travaux Pratiques and Projects at Observatoire de Paris. For more information, contact Pierre Baudoz.

## Windows Installation

- Download VimbaX : https://www.alliedvision.com/en/products/software/vimba-x-sdk/#c13326

	You can download the latest version of VimbaX SDK from the link above. The version used for this project is `VimbaX_2023.4`. After downloading this file, run the installer: https://downloads.alliedvision.com/VimbaX/VimbaX_Setup-2023-4-Win64.exe

At this point, you have two options: 

- using the recompiled binaries : (Not available yet)
- compiling the binaries yourself

This next section will guide you to compile the binaries yourself. If you want to use the recompiled binaries, skip to the next section.

- Install the windows build tools: https://aka.ms/vs/17/release/vs_BuildTools.exe

	with the following components:
	- MSVC v143 - VS 2022 C++ x64/x86 build tools
	- C++ CMake tools for Windows
	- C++ Build Tools core features
	- C++ core features
	- C++ Clang Compiler for Windows
	- Windows 11 SDK

- Download GIT : https://git-scm.com/download/win
- Download Anaconda : https://www.anaconda.com/download/success and install it with this default options

  If you have issues with powershell on conda init: please change the default shell to bash by running the following command in powershell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  Then, open a new powershell window and run the following command:

- Compile the catkit2 platform
    - Open a `powershell` terminal and run the following commands:
	```powershell
	git clone https://github.com/spacetelescope/catkit2
	cd catkit2

	git checkout fcce243
	cd extern
	./download.sh
	cd ..
	conda env create --file environment.yml
	conda activate catkit2
	pip install -e .
	cd ..
	```

- Retrieve the ssh keys in order to clone the TP controller repository.
	- Run the following commands in the `powershell` terminal:
	```powershell
	New-Item -ItemType Directory -Force -Path $HOME\.ssh
	scp -r login@datahra:/volumes/hra/thd/keys_thd/id_rsa_thd_user* $HOME\.ssh\
	icacls $HOME\.ssh\* /inheritance:r /grant:r "$($env:USERNAME):(F)"
	git config --global core.sshCommand "ssh -i $HOME\.ssh\id_rsa_thd_user"
	```

- Install the TP controller
	- Run the following commands in the `powershell` terminal:
	```powershell
	git clone git@github.com:pbaudoz/TP_OASE.git
	cd TP_controller
	conda env update --file environment.yml --name 'catkit2'
	pip install -e .
	cd ..
	```

	You can add the following line to your `.bashrc` file to activate the environment automatically when you open a terminal:

	```bash
	echo "conda activate catkit2" >> ~/.bashrc
	```

## Linux Installation from sources

### Install dependencies

```bash
apt update 
apt install build-essential xauth git curl wget
```

### Install Vimba SDK (semble indispensable => don't install VimbaX?)

```bash
wget https://downloads.alliedvision.com/Vimba64_v6.0_Linux.tgz
tar -xvf Vimba64_v6.0_Linux.tgz
sudo ./Vimba_6_0/VimbaUSBTL/Install.sh
```

### Install VimbaX SDK

```bash
wget https://downloads.alliedvision.com/VimbaX/VimbaX_Setup-2024-1-Linux64.tar.gz
tar -xvf VimbaX_Setup-2024-1-Linux64.tar.gz
sudo ./VimbaX_2024-1/cti/Install_GenTL_Path.sh
```

### Retrieve the ssh keys
(not working)
```bash
mkdir -p ~/.ssh
scp -r login@datahra:/volumes/hra/thd/keys_thd/id_rsa_thd_user* .ssh/
chmod -R 600 ~/.ssh
git config --global core.sshCommand 'ssh -i $HOME/.ssh/id_rsa_thd_user'
```

**NOTE**: The ssh keys are stored in the datahra server. You need to have access to the datahra server to retrieve the keys.

### Install Miniconda with python3

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Install the catkit2 platform

```bash
git clone https://github.com/spacetelescope/catkit2
cd catkit2
git checkout fcce243
cd extern
./download.sh
cd ..
conda env create --file environment.yml
conda activate catkit2
python setup.py develop
cd ..
```

### Install the TP_OSAE

```bash
git clone git@github.com:pbaudoz/TP_OSAE.git
cd TP_OSAE
conda env update --file environment.yml --name 'catkit2'
pip install -e .
cd ..
```

### Install vmbpy

```bash
wget https://github.com/alliedvision/VmbPy/releases/download/1.0.5/vmbpy-1.0.5-py3-none-any.whl
pip install vmbpy-1.0.5-py3-none-any.whl
```

### Install g++-11

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11
```

### Configure your environment

Add to your .bashrc:

```bash
conda activate catkit2
```

This note seems optional (see if it's still useful): 

> Add a CATKIT2_ROOT_DIR environment variable that points to the root of the catkit2 repo directory. 
> This is necessary for the C++ compiler to find the catkit_core library and include files.
> 
> On MacOS, assuming a bash shell, this is done by adding the following line at the end of your .bash_profile file:
> ```
> export CATKIT2_ROOT_DIR='path_to_catkit2_root_directory'
> ```
> The .bash_profile file is located in your home directory and can be created if it does not exist.
> This line can be added using VI or a text editor like BBEdit that allows to edit hidden files.

### Configure the controller service

edit the file `~/TP_OSAE/tposae/config/services.yml`:

and modify the `camera_id`, `device_name`, `sensor_width`, `sensor_height` in the `detector` section to match your system.

You can try to find the correct values by running the following command: (not tested)

```bash
~/VimbaX_2024-1/bin/ListCameras_VmbCPP
```
### To test without runnning the gui (Optionnal)
Help debugging but sometimes the test is not working but the gui is working...

#### Start the controller server
In a first terminal

```bash
tposae start server
```
#### Run the test
In a second terminal, go to the catkit2/catkit2/services/alliedvision

```bash
python allied_vision_camera.py --id camera --port 5222 --testbed_port 2345
```

## Usage

### Start the controller server

```bash
tposae start server
```

### Start a .py files example like 
    
```bash
controller start gui
```

