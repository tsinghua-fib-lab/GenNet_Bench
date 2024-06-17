# pycomm

Python reinforcement learning client for communication with the simulator.

## Usage

### Initialization: (Using Dynamic Optimization Environment as an Example)

```
from pycomm.pycomm import CoverageEnv
env = CoverageEnv(
    job: str = 'job0',            # Job name
    coverage_range: int = 1000,  # Coverage range
    handover_interval: int = 1,   # Handover interval
)
```


### Simulator Step: Control the simulation steps of the simulator

#### Parameters:

 - action: Actions provided by RL (Reinforcement Learning)
 
 - start: Starting step of the simulator

 - total: Total number of steps the simulator should execute (end step = starting step + total steps)

 - interval: Time interval for each step

 - is_output: bool, whether to save visualization content.

 
#### Return Values:

 - async_coroutine_step: Contains state and reward
 
 - state: State returned by the simulator, used for RL optimization training
 
 - reward: Result of taking action in the simulator, providing reward feedback 
 - visualize: Visualization Content
 
 - terminated: Indicates when the simulation is finished
 
```
	async_coroutine_step, terminated, _ = await env.step(action, start, total, interval, is_output )
```

### Closing the Simulator: Terminate the simulator process when RL training is complete.
```
	 await env.reset()
```

# MORL Algorithm Interaction with Simulator

In this scenario, we divide the Beijing Guomao area into grids of 10m x 10m each. Using a channel model, we calculate the signal strength for each grid with respect to all base stations. We select the base station with the strongest signal for each grid and determine if the signal strength exceeds a predefined threshold. If it does, the grid is considered covered; otherwise, it is not covered. The objective of base station coverage optimization is to maximize the coverage of all grids and the total user throughput. The configuration of base station azimuth, downtilt, and beamwidth affects the propagation and reception quality of base station signals, thus impacting the coverage area. For example, when the base station's azimuth is small, the signal mainly propagates in the direction of the base station, leading to a smaller coverage area. On the other hand, a larger base station azimuth can result in signals propagating in a wider range but may also lead to signal dispersion and increased interference, resulting in weaker signal strength. The coverage optimization task involves optimizing the angles and beamwidth configurations of base stations to simultaneously improve the coverage rate and user total throughput. The state information for this task includes the status of each grid in the system, including signal strength from all base stations and whether it is indoors or outdoors. The reward function represents the reward signal received by the simulation engine based on feedback from different strategies, i.e., whether it is covered and the average communication rate per user. Based on the returned state and reward function, the network optimizer adjusts the azimuth, downtilt, and beamwidth configurations of base stations to optimize both the overall coverage rate and user total throughput of the grid.

The simulator transfers the current environment state (State) to the coverage optimization RL algorithm, which outputs actions (Action) and transfers them back to the simulator. The process continues in a loop until a termination state or condition is reached. The three elements of the coverage optimization RL algorithm, State, Action, and Reward, have the following structure:

State: [{"grid_id": xxx, "signals": [xxx], "user_number": xxx, "is_indoor": true/false}] (Latest status for a given time period T)

Action: [{"id": xxx, "bs_azi": xxx, "bs_tilt": xxx, "power": xxx}]

Reward: [{"grid_id": xxx, "coverage": true/false, "mean_rate": xxx}] (Mean rate over time period T)


The meanings of each field are as follows:

State——

grid_id: Grid identifier

signals: Signal strength from each base station to the grid

user_number: Number of users in the grid (varies over time)

is_indoor: Indicates whether it is indoor (affects signal strength)

Action——

id: Base station (antenna) identifier

bs_azi: Azimuth of the antenna

bs_tilt: Downtilt of the antenna

power: Power of base station

Reward——

grid_id: Grid identifier

coverage: Indicates whether it is covered

mean_rate: Mean rate over time period T (rate is related to received signal strength)

