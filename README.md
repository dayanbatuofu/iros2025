# Parametric Value Approximation for General-sum Differential Games with State Constraints
<br>
Lei Zhang,
Mukesh Ghimire, 
Wenlong Zhang, 
Zhe Xu, 
Yi Ren<br>
Arizona State University

This is our IROS paper: <a href="https://arxiv.org/pdf/2401.01502"> "Parametric Value Approximation for General-sum Differential Games with State Constraints"</a>

## Get started
There exists two different environment, you can set up a conda environment with all dependencies like so:

For Narrow road collision avoidance/Double lane change/Two drone collision avoidance
```
conda env create -f environment.yml
conda activate siren
```
For BVP_generation
```
conda env create -f environment.yml
conda activate hji
```

## Code structure
There are two folders with different functions
### BVP_generation: use standard BVP solver to collect the Nash equilibrial (NE) values for narrow_road_collision_avoidance/double_lane_change/two_drone_collision_avoidance
The code is organized as follows:
* `generate_narrow_road.py`: generate 9D NE values functions for case 1 under player type configurations $\Theta^2$.
* `generate_lane_change.py`: generate 9D NE values functions for case 2 under player type configurations $\Theta^2$.
* `generate_drone_avoidance.py`: generate 13D NE values functions for case 3 under player type configurations $\Theta^2$.
* `./utilities/BVP_solver.py`: BVP solver.
* `./example/vehicle/problem_def_narrow_road_nop.py`: dynamic, PMP equation setting for case 1.
* `./example/vehicle/problem_def_lane_change_nop.py`: dynamic, PMP equation setting for case 2.
* `./example/vehicle/problem_def_drone_avoidance_nop.py`: dynamic, PMP equation setting for case 3.

run `generate_narrow_road_nop.py`, `generate_lane_change_nop.py`, `generate_drone_avoidance_nop.py`, to collect NE values. Please notice there are player type configurations $\Theta^2$ for narrow road collision avoidance/double lane change/two drone collision avoidance. Data size can be set in `./example/vehicle/problem_def_narrow_road_nop.py`, `./example/vehicle/problem_def_lane_change_nop.py`, `./example/vehicle/problem_def_drone_avoidance_nop.py`

### HJ_NO_Narrowroad(narrow_road_collision_avoidance)
The code is organized as follows:
* `dataio.py`: load training data for HNO and SNO.
* `training_hno.py`: contains HNO training routine.
* `training_sno.py`: contains SNO training routine.
* `loss_functions.py`: contains loss functions for HNO and SNO.
* `modules_hno.py`: contains HNO architecture.
* `modules_sno.py`: contains SNO architecture.
* `utils.py`: contains utility functions.
* `diff_operators.py`: contains implementations of differential operators.
* `./experiment_scripts/train_intersection_HNO.py`: contains scripts to train the HNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./experiment_scripts/train_intersection_SNO.py`: contains scripts to train the SNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./validation_scripts/closedloop_traj_generation_hno_tanh.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_tanh_sym.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_sine.py`: use HNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_sine_sym.py`: use HNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_relu.py`: use HNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_relu_sym.py`: use HNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_tanh.py`: use SNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_tanh_sym.py`: use SNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_sine.py`: use SNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_sine_sym.py`: use SNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_relu.py`: use SNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_relu_sym.py`: use SNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/NTK_analysis_tanh.py`: compute condition numbers of neural tangent kernel for HNO with tahn.
* `./validation_scripts/NTK_analysis_sine.py`: compute condition numbers of neural tangent kernel for HNO with sin.
* `./validation_scripts/NTK_analysis_relu.py`: compute condition numbers of neural tangent kernel for HNO with relu.
* `./validation_scripts/model`: experimental model in the paper.
* `./validation_scripts/train_data`: training data in the paper.
* `./validation_scripts/test_data`: testing data in the paper. Download the test data: <a href="https://drive.google.com/drive/folders/1JvLFhIn9lb_oNtfpTkk6yPh3fNQpHL3t?usp=sharing"> link.

### HJ_NO_Lanechange(double_lane_change)
The code is organized as follows:
* `dataio.py`: load training data for HNO and SNO.
* `training_hno.py`: contains HNO training routine.
* `training_sno.py`: contains SNO training routine.
* `loss_functions.py`: contains loss functions for HNO and SNO.
* `modules_hno.py`: contains HNO architecture.
* `modules_sno.py`: contains SNO architecture.
* `utils.py`: contains utility functions.
* `diff_operators.py`: contains implementations of differential operators.
* `./experiment_scripts/train_intersection_HNO.py`: contains scripts to train the HNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./experiment_scripts/train_intersection_SNO.py`: contains scripts to train the SNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./validation_scripts/closedloop_traj_generation_hno_tanh.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_tanh_sym.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_sine.py`: use HNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_sine_sym.py`: use HNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_relu.py`: use HNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_relu_sym.py`: use HNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_tanh.py`: use SNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_tanh_sym.py`: use SNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_sine.py`: use SNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_sine_sym.py`: use SNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_relu.py`: use SNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_relu_sym.py`: use SNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/NTK_analysis_tanh.py`: compute condition numbers of neural tangent kernel for HNO with tahn.
* `./validation_scripts/NTK_analysis_sine.py`: compute condition numbers of neural tangent kernel for HNO with sin.
* `./validation_scripts/NTK_analysis_relu.py`: compute condition numbers of neural tangent kernel for HNO with relu.
* `./validation_scripts/model`: experimental model in the paper.
* `./validation_scripts/train_data`: training data in the paper.
* `./validation_scripts/test_data`: testing data in the paper. Download the test data: <a href="https://drive.google.com/drive/folders/1JvLFhIn9lb_oNtfpTkk6yPh3fNQpHL3t?usp=sharing"> link.

### HJ_NO_Drone(two_drone_collision_avoidance)
The code is organized as follows:
* `dataio.py`: load training data for HNO and SNO.
* `training_hno.py`: contains HNO training routine.
* `training_sno.py`: contains SNO training routine.
* `loss_functions.py`: contains loss functions for HNO and SNO.
* `modules_hno.py`: contains HNO architecture.
* `modules_sno.py`: contains SNO architecture.
* `utils.py`: contains utility functions.
* `diff_operators.py`: contains implementations of differential operators.
* `./experiment_scripts/train_intersection_HNO.py`: contains scripts to train the HNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./experiment_scripts/train_intersection_SNO.py`: contains scripts to train the SNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./validation_scripts/closedloop_traj_generation_hno_tanh.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_tanh_sym.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_sine.py`: use HNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_sine_sym.py`: use HNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_relu.py`: use HNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_relu_sym.py`: use HNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_tanh.py`: use SNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_tanh_sym.py`: use SNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_sine.py`: use SNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_sine_sym.py`: use SNO (sin as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_relu.py`: use SNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_sno_relu_sym.py`: use SNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/NTK_analysis_tanh.py`: compute condition numbers of neural tangent kernel for HNO with tahn.
* `./validation_scripts/NTK_analysis_sine.py`: compute condition numbers of neural tangent kernel for HNO with sin.
* `./validation_scripts/NTK_analysis_relu.py`: compute condition numbers of neural tangent kernel for HNO with relu.
* `./validation_scripts/model`: experimental model in the paper.
* `./validation_scripts/train_data`: training data in the paper.
* `./validation_scripts/test_data`: testing data in the paper. Download the test data: <a href="https://drive.google.com/drive/folders/1JvLFhIn9lb_oNtfpTkk6yPh3fNQpHL3t?usp=sharing"> link.

## Contact
If you have any questions, please feel free to email the authors.

