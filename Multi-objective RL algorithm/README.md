# pycomm

Python reinforcement learning client for communication with the simulator.

## Usage

### Initialization: 

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
	async_coroutine_step, terminated, _ = env.step(action, start, total, interval, is_output )
```

### Closing the Simulator: Terminate the simulator process when RL training is complete.
```
        env.reset()
```

# MORL Algorithm Interaction with Simulator


The simulator transfers the current environment state (State) to the MORL algorithm, which outputs actions (Action) and transfers them back to the simulator. The process continues in a loop until a termination state or condition is reached. The three elements of the MORL algorithm, State, Action, and Reward, have the following structure:

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

