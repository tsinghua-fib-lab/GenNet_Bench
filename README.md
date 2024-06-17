# GenNet_Bench

# Welcome to the GenNet Repository

This is the implement of 

# File Description

This repository includes the following key components:

## GenNet - Interactive Simulator

GenNet is an interactive simulator designed to generate a Synthetic Data Ecosystem for Mobile Communication Networks. It allows researchers and developers to emulate mobile network scenarios and experiment with various parameters. In this release, you will find:

- **the GenNet Executable (GenNet):** The core executable file for running the GenNet simulator. (https://github.com/tsinghua-fib-lab/GenNet_scientific_data/releases/download/v1.0.1/GenNet)

## pycomm

- **pycomm:** A Python reinforcement learning client (https://github.com/tsinghua-fib-lab/GenNet_scientific_data/tree/main/pycomm) that facilitates communication with the simulator, enabling dynamic interactions and experiments.

## Milti-objective Reinforcement Learning Optimization Codes

We provide three sets of multi-objective reinforcement learning optimization codes (https://github.com/tsinghua-fib-lab/GenNet_scientific_data/tree/main/Reinforcement%20Learning%20Optimization%20Codes) tailored to enhance the performance of the GenNet simulator in different ways


## README Documentation

For detailed instructions on how to get started, configure, and use GenNet, please refer to the README documentation provided in this repository.


# Utilization of GenNet

GenNet is a powerful software tool that serves as "A Generative AI-Powered Synthetic Data Ecosystem for Mobile Networks." This part provides a comprehensive guide on how to use GenNet, including startup parameters with examples and detailed explanations for all possible parameter values in the configuration file.

## Startup Parameters and Their Meanings

GenNet accepts several startup parameters that control its behavior. Here are the main parameters with examples and their meanings:

```bash
Usage of ./comm:
  -config string
        config file path
  -decode.gc_ignore_time duration
        decode: GC (Garbage Collection) ignores time, and data accessed from the last visit to the current time within this period will not be GC'd (default 5m0s).
  -decode.gc_reserved_count int
        decode: GC retention count, the most recent few items of data will not be GC'd (default 5).
  -decode.gc_start_threshold int
        decode: The minimum cache count to trigger GC (default 10).
  -job string
        the name of the whole simulation task (default "job0")
  -listen string
        gRPC listening address (default "localhost:51402")
  -pprof string
        pprof listening address
  -rl
        use RL rather than normal simulation
```

To run the program, you need to provide the necessary command-line arguments based on your requirements. Here's an example command to run the program:

```
./comm -config /path/to/config/file -job job0 -listen localhost:51402
```

Make sure to replace /path/to/config/file with the actual path to your configuration file if required. Additionally, you can adjust the values of the command-line arguments according to your specific needs.

Note: The command assumes that the GenNet executable file is located in the current directory. If it's in a different directory, you' ll need to provide the correct path to the executable file.

## How to Use GenNet

To use GenNet, follow these steps:

1. **Download Files**: Make sure to place the complete 'pycomm' folder https://github.com/tsinghua-fib-lab/GenNet_scientific_data/releases/download/v1.0.1/comm, 'data' folder, the configuration file https://github.com/tsinghua-fib-lab/GenNet_scientific_data/releases/download/v1.0.1/update_config.yml, and the optimization code (user's own code) in the main directory. (The 'pycomm' folder should be located within a subdirectory of the optimization code.)

2. **Start the Simulator**: Open a terminal and navigate to the directory where GenNet is located. Start the emulator with the following command: ./GenNet -config /path/config.yml -job test -rl.

3. **Start Interaction**: Open a new terminal and activate the Conda environment to initiate the RL interaction.

## Precautions

1. If you are using a Docker virtual machine, you need to install version 15.0 of the libprof library (Python module).

2. The data files should be downloaded in advance and placed in the main directory. Data file address: https://cloud.tsinghua.edu.cn/f/7ae439a5ece044bea6ae/ .

3. The "pycomm" folder should be located in a subdirectory of the optimization algorithm code for interaction.

## Configuration File Options

The `config.yml` file contains various configuration options. Here are detailed explanations for all possible parameter values:

- `ControlStep`: Defines the starting step, total steps, and step interval.
  - `start` (int32): Specifies the starting point for control steps.
  - `total` (int32): Defines the total number of control steps.
  - `interval` (double): Sets the time interval for control steps.

- `ControlThread`: Defines the number of threads.
  - `worker` (int32): Specifies the number of worker threads for control.

- `ControlOutput`: Defines the latitude and longitude of the area of interest.
  - `min_longitude` (double): Defines the minimum longitude value for the control output.
  - `min_latitude` (double): Specifies the minimum latitude value for the control output.
  - `max_longitude` (double): Sets the maximum longitude value for the control output.
  - `max_latitude` (double): Specifies the maximum latitude value for the control output.

- `ChannelType`: Specifies the type of channel used in the simulation. Possible values:
  - `CHANNEL_TYPE_EQUATION` (1)    // Based on ray tracing
  - `CHANNEL_TYPE_RAY_TRACING` (2)
  - `CHANNEL_TYPE_3GPP` (3)
  - `CHANNEL_TYPE_FREESPACE` (4)    //	Based on electromagnetic computation.
  - `CHANNEL_TYPE_CEM` (5)

- `AntennaType`: Specifies the type of antenna used in the simulation. Possible values:
  - `ANTENNA_TYPE_SISO` (1)
  - `ANTENNA_TYPE_MIMO` (2)

- `Control`: Controls various aspects of the simulation, such as steps, threads, and display options.
  - `step` (ControlStep): Specifies the control step.
  - `thread` (ControlThread): Specifies the control thread.
  - `microscopic_range` (ControlOutput): Defines the range of the microscopic area.
  - `macroscopic_range` (ControlOutput): Defines the range of the macroscopic area.
  - `enable_controlled` (bool): Indicates whether the simulator uses external control.
  - `enable_optimize` (bool): Specifies if communication uses optimized allocation.
  - `optimize_interval` (optional int32): Sets the communication optimization allocation interval in steps.
  - `display_guomao` (bool): Determines whether to display the microscopic area.
  - `coverage_range` (double): Sets the initial coverage range of base stations in meters.
  - `handover_interval` (int32): Defines the handover frequency of base stations in seconds.
  - `channel_type` (ChannelType): Specifies the channel model used in the simulation.
  - `antenna_type` (AntennaType): Specifies the type of antenna used in the simulation.

- `mongo`: Specifies Mongo DB
  - `uri`: MongoDB address
  - `map`: Map Data
  - `comm_topo`: Communication Topology
  - `comm_demand`: Communication Demand
  - `ray_tracing_loss`: Communication Ray Tracing

For detailed parameter usage examples, please refer to pycomm/pycomm/config.yaml.

## Database Dependencies

GenNet relies on a MongoDB database for its functionality. Make sure you have MongoDB installed and configured properly.

**Note**: Ensure that you have all the necessary dependencies and permissions to run GenNet effectively.

