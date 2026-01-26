## How to Install MatchFlow on a MacOS M1 Machine

This is a step-by-step guide to install MatchFlow on a single macOS machine with an M1 chip. If you are unsure whether your Mac has an M1 chip, click on the Apple in the top left corner of your screen > About This Mac. If it says Chip Apple M1, then you have an M1 chip. If it does not say Chip Apple M1, you do not have an M1 chip.

This guide has been tested on a 2020 MacBook Pro with an Apple M1 Chip, 8GB Memory, macOS version Sequoia 15.0.1, and a .zshrc profile, using Python 3.12. You can try to adapt this guide to other configurations.

If your machine has an Intel, M2, M3, or M4 chip, this installation guide may work for you, but we have not tested it.

Once you have installed MatchFlow on a single macOS machine, you can run it using Pandas or Spark.

### Step 1: Installing Python

We first install Python 3.12, create a virtual environment, and then install two Python packages setuptools and build. Other versions of Python, other environments, or incorrect installations of the setuptools and build packages can cause issues with MatchFlow installation.

If you suspect that you may have Python downloaded on your machine already, open up your terminal. Then, run the command:

    which python

If the output path says

“/usr/local/bin/python”

run:

    python --version

If the output of this is

“Python 3.12.x”

where x is a number, you can go to Step 1C after completing Step 1A (you do not need to complete Step 1B).

If

which python

Or

python --version

do not have the outputs listed above, do all substeps A-D.

#### Step 1A: Installing Homebrew

Before installing Python, we need to ensure we have Homebrew installed. Homebrew is a popular open-source package manager for macOS and Linux, used to simplify installing, updating, and managing software and libraries.

To check if Homebrew is installed, open up a terminal. Then, type

```
brew info
```

If the output contains kegs, files, and GB, then Homebrew is installed and you can go to Step 1B. Otherwise, you need to install Homebrew.

To install Homebrew, run the following command in your terminal:

```
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com](https://raw.githubusercontent.com)
Homebrew/install/HEAD/install.sh)"
```

The installation may prompt you to follow further instructions to add this to your PATH variables; if so, follow those onscreen instructions. This will make it easier to use brew later. If you see

“Installation Successful!”

in the output, you are done with this step.

#### Step 1B: Installing Python 3.12

To download Python environments, we will use Homebrew. Run the following in the terminal to install Python 3.12:

```
	brew install python@3.12
```

#### Step 1C: Setting Up the Python Environment

Now we will create a Python environment with Python 3.12. This step is necessary to make sure we use the correct version of Python with the correct dependencies. In your terminal, run:

```
	python -m venv ~/matchflow-venv
```

This will create a virtual environment named matchflow-venv. To activate this environment, run the following:

```
	source ~/matchflow-venv/bin/activate
```

To make sure everything is correct, run:

```
	python --version
```

If the output says

“Python 3.12.x”

where x ≥ 0, then the Python environment setup is successful.

#### Step 1D: Installing Python Packages setuptools and build

Before installing these packages, make sure you are in the virtual environment. If you have just finished Step 1C, you are in the virtual environment. Otherwise, to make sure your virtual environment is active, you can run:

```
	source ~/matchflow-venv/bin/activate
```

To install setuptools, run:

```
	pip install setuptools
```

To install build, run:

```
	pip install build
```

If at any point during the installation you close your terminal, you will need to reactivate your virtual environment by running:

```
	source ~/matchflow-venv/bin/activate
```

### Step 2: Installing Java

We strongly recommend installing Java Temurin JDK 17, which is a specific Java release that we have extensively experimented with. We will use Homebrew for the installation. If you do not use Homebrew, you may download a version of Java that is not supported by an arm64 architecture. If you do not have Homebrew installed, or are unsure if you have Homebrew installed, please look at step 1A above to install Homebrew.

#### Step 2A: Java Installation

We are going to download the Temurin 17 JDK that is optimized for your machine running on an M1 chip. To install Java, open a terminal and run the following:

```
	brew install --cask temurin@17
```

To check if this was successful, run the following command:

```
	/usr/libexec/java_home -V
```

If the output contains

```
“OpenJDK version 17.x.y”
```

where x and y are numbers, the installation was successful.

#### Step 2B: Setting JAVA_HOME Environment Variable

We need to set this version of Java as the default for our machine. To do so, open up a terminal and make sure you are in the root directory (if you are unsure if you are in the root directory or not, run cd to get back to the root directory). Next, run:

```
	echo 'export JAVA_HOME=$(/usr/libexec/java_home -v17)' >> ~/.zshrc
```

To enact these changes, run:

```
	source ~/.zshrc
```

To check if these changes were successful, run:

```
	echo $JAVA_HOME
```

If the installation was successful, you will see a file path output.

### Step 3: Installing MatchFlow

In the future you can install MatchFlow using one of the following two options. **As of now, since MatchFlow is still in testing, we do not yet enable Option 1 (Pip installing from PyPI). Thus you should use Option 2 (Pip installing from GitHub).**

#### Option 1: Pip Installing from PyPI

**Note that this option is not yet enabled. Please use Option 2.**

You can install MatchFlow from PyPI, using the following command:

```
	pip install MatchFlow
```

This command will install MatchFlow and all of its dependencies, such as Flask, Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Pyarrow, Py_Stringmatching, PySpark, Requests, Scikit-Learn, Scipy, Streamlit, Tabulate, Threadpoolctl, TQDM, Xgboost, Xxhash.

#### Option 2: Pip Installing from GitHub

Instead of pip installing from PyPI, you may want to pip install MatchFlow from its GitHub repo. This happens if you want to install the latest MatchFlow version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

To install MatchFlow directly from its GitHub repo, use the following command:

    pip install git+https://github.com/MadMatcher/MatchFlow.git@main

Similar to pip installing from PyPI, the above command will install MatchFlow and all of its dependencies.
