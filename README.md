<br>
<br>
<br>

<div align="center">
  <img src="assets/introduction.png">
</div>


<div align="center">
  <h1>
  PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration
</h1>
</div>


<div align="center">

[Mellen Y. Pu](https://dandelionym.github.io/)
&emsp;&emsp;&emsp;
[Tao Lin]()
&emsp;&emsp;&emsp;
[Hongyu Chen]()
 
Westlake University

</div>


<div align="center">
  <p>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-CC BY NC 4.0-yellow.svg" alt="License: MIT">
    </a>
    &emsp;
    <a href="">
      <img src="https://img.shields.io/badge/AI4SD-Fully Adaptable & Generalizable-blue.svg" alt="License: MIT">
    </a>
    &emsp;
    <a href="https://arxiv.org/abs/2505.15047">
      <img src="https://img.shields.io/badge/arXiv-2505.15047-red.svg" alt="License: MIT">
    </a>
  </p>
</div>


## 👋 Overview
We introduce `PiFlow`, an information-theoretical framework. It uniquely treats automated scientific discovery as a structured uncertainty reduction problem, guided by foundational principles (e.g., scientific laws). This ensures a more systematic and rational exploration of scientific problems.

:ballot_box_with_check: You can directly use our PiFlow for **ANY** of your specific tasks for assisting scientific discovery!


## 📃 Results
PiFlow has demonstrated significant advancements in scientific discovery:
* Evaluated across three distinct scientific domains:
    * 🔬 Nanohelix.
    * 🧬 Bio-molecules.
    * ⚡ Superconductors.
* Markedly improves discovery efficiency, reflected by a **73.55% increase** in the Area Under the Curve (AUC) of property values versus exploration steps.
* Enhances solution quality by an impressive **94.06%** compared to a vanilla agent system.


<div align="center">
  <img src="assets/results.png" alt="results">
</div>


PiFlow serves as a Plug-and-Play method, establishing a novel paradigm shift in highly efficient automated scientific discovery, paving the way for more robust and accelerated AI-driven research. Our PiFlow accommodates various scenarios (bio-molecules, nanomaterials and superconductors discovery) with experimental conditions (i.e., tools for agent), necessitating little to no prompt engineering for effective agent-level interaction.

## 🔧 Setup and Run

### 0. Install

To prepare the environment, we recommend executing the following conda instructions: 

```shell
git clone https://github.com/amair-lab/PiFlow && cd PiFlow
conda create -f environment.yml  # The environ name will be `piflow`
conda activate piflow 

# We suggest to use mamba for faster installation, though conda works fine, but for preferences... well
# If you want to use mamba for package management, run these commands:
# conda install mamba -n base -c conda-forge
# mamba env create -f environment.yml
# mamba activate piflow
```


### 1. Launch Dynamic Environment

We have developed three types of experiments named `AgenX...` (e.g., `AgenX_Chembl35`).
To open the `launch.py` for each scenario's task, run:

```shell
python launch.py
````

### 2. Prepare Your API Key

You need to apply for any **OpenAI compatible API KEYs** for calling any models. Ensure the model embedded at the experiment agent is able to use tools.


### 3. About Configs and Demo

You can first configure the running commands in the `/configs/` directory, or simply try the demo after model configuration.

In the config for model, every agent contains the config of API and its display mode:

```yaml
        streaming:  # bool, true means the output will be in streaming, the same with the original API output
        api_config:
            base_url:   # Follow the OpenAI API
            model_name: # Follow the OpenAI API
            is_reasoning:   # bool, ture if it is a reasoning model like o1, R1, QwQ, etc. (typically has <think></think> token)
            api_key:    # Follow the OpenAI API
            temperature:    # Follow the OpenAI API, generally we set 0.6 or 0.4
            max_tokens:     # We set 4096 here
        tools: []   # a list of the tools name, it should be contained and claimed in the `tools/__init__.py`, and follows the YAML's list fashion
```

Other field like `enabled` means if this Agent is a member of group chatting. Fields like `name` and `description` are the same to the building of Agent profile. Note that, the `name` field should not contain any blank char. 

After configuration, you can run the demo scenario of nanohelix material discovery, by running:

```shell
bash ./run_demo.sh
```

You will see the detailed output with colored chatting history at the command line. Our philosophy is, one config file corresponds to one task, and you should prepare both task config and model config for running PiFlow. 



## 🪄 Adapt to Your Own Task

There are many possible tasks that we can use PiFlow to assist the discovery processes: 

<div align="center">
<table style="margin-left: auto; margin-right: auto;">
  <tr>
    <td style="padding: 5px; text-align: center;">
      <img src="assets/quantums.jpeg" alt="Description of Image 3" style="width: 200px;">
      <div style="text-align: center">&emsp;&emsp;Quantum Materials</div>
    </td>
    <td style="padding: 5px; text-align: center;">
      <img src="assets/battery.jpeg" alt="Description of Image 1" style="width: 200px;">
      <p><em>&emsp;&emsp;&emsp;&emsp;Battery</em></p>
    </td>
    <td style="padding: 5px; text-align: center;">
      <img src="assets/protein.jpeg" alt="Description of Image 2" style="width: 200px;">
      <p><em>&emsp;&emsp;&emsp;&emsp;Proteins</em></p>
    </td>
  </tr>
</table>
</div>

PiFlow offers full flexibility to adapt to your own scenarios, such as quantum science, MOF synthesis, and others. To adapt to your task, you could design tools based on the given examples at `src/tools/`, simply copy one file, e.g., `_nanohelix_tools.py` and create new one named be your scenario (e.g., `_mof_tools.py`). 

### 1. Create your own tools
Replace the tools with your own preferred one (Http tools or others), and finally import the tool into `src/tools/__init__.py`. The PiFlow will automatically find your tools and register it. 

### 2. Configurations
Don't forget to write the configuration file in the `/config/`: For models, just keep it and fill your own API and add tool name to the experiment agent, and for tasks, you should formally define your task similar to the given examples. 

### 3. Test tools and run
Remember to test your tools to make sure it works fine. After that, you could run with command like `run_PiFlow.sh` and ANY prompt engineering are NOT needed for each agent. 


## 📚 Citation
```bibtex
@misc{pu2025piflow,
      title={PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration}, 
      author={Yingming Pu and Tao Lin and Hongyu Chen},
      year={2025},
      eprint={2505.15047},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.15047}, 
}
```


## 📄 License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/). Under this license, you are free to share and adapt this work for non-commercial purposes, provided you give appropriate attribution.
