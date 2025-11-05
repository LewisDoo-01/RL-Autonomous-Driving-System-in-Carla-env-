# Autonomous Driving in CARLA using Deep Reinforcement Learning

This project implements an autonomous driving agent in the [CARLA](http://carla.org/) simulator using deep reinforcement learning. The agent is trained to navigate through a town, avoiding obstacles and following the road.

## Project Structure

```
├── autoencoder/            # Variational Autoencoder for image processing
├── carla/                  # CARLA simulator Python client
├── checkpoints/            # Saved model checkpoints
├── networks/               # PPO and DDQN network models
├── preTrained_models/      # Pre-trained models
├── runs/                   # Tensorboard logs
├── simulation/             # CARLA simulation environment setup
├── continuous_driver.py    # Main script for PPO agent
├── discrete_driver.py      # Main script for DDQN agent
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Autonomous_Driving
    ```

2.  **Install dependencies:**
    It is recommended to use a conda environment.
    ```bash
    conda create --name autonomous_driving python=3.7
    conda activate autonomous_driving
    ```
    Install the required packages using:
    ```bash
    conda install future==0.18.3
    conda install numpy==1.21.1
    conda install pygame==2.1.2
    conda install poetry==9.4.0
    conda install poetry==1.3.2
    ```

3.  **Download CARLA:**
    Download CARLA simulator version 0.9.8 from the official website: [https://github.com/carla-simulator/carla/releases/tag/0.9.8](https://github.com/carla-simulator/carla/releases/tag/0.9.8)

## Usage

1.  **Start the CARLA simulator:**
    For low quality settings, run:
    ```bash
    "C:\path\to\CARLA_0.9.8\WindowsNoEditor\CarlaUE4.exe" -quality-level=Low -ResX=800 -ResY=600
    ```
    Replace `"C:\path\to\CARLA_0.9.8"` with the actual path to your CARLA installation.

2. **Run the agent:**
    Open a new terminal, activate the conda environment and navigate to the project directory.

    *   **For the PPO agent (continuous control):**
        ```bash
        python continuous_driver.py --exp-name ppo --town Town07 train False
        ```
        
    *   **For the DDQN agent (discrete control):**
        ```bash
        python discrete_driver.py --exp-name ddqn --town Town07 train False
        ```

![demo](https://github.com/user-attachments/assets/59a9902f-536c-498f-8eb3-c7ac28a7b764)


3.  **Train the agent:**
    Open a new terminal, activate the conda environment and navigate to the project directory.

    *   **For the PPO agent (continuous control):**
        ```bash
        python continuous_driver.py --exp-name ppo --town Town07
        ```
        
    *   **For the DDQN agent (discrete control):**
        ```bash
        python discrete_driver.py --exp-name ddqn --town Town07
        ```
        
    You can add
    ```bash
    python discrete_driver.py --exp-name ddqn --town Town07 --load-checkpoint True
    ```
    if you want to load checkpoint from the lastest training.

4.  **Monitor training with Tensorboard:**
    ```bash
    tensorboard --logdir=runs
    ```

## Additional Notes

*   The `parameters.py` file contains various hyperparameters for the models and the simulation.
*   The `autoencoder` is used to compress the input images from the simulator.
*   The project uses both PPO (on-policy) and DDQN (off-policy) reinforcement learning algorithms.

