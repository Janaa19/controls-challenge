# My Solution — Jana Abedeljaber

Hi! This is my solution to the comma.ai Controls Challenge. The original README from the repo is below this explanation.

Final Score: **46.92** (5000 segments)  
Baseline PID: **84.85**

Total compute for optimization was about **5 days on an 8-core MacBook Pro**, running roughly **8 million simulator rollouts**.

My approach to this challenge was mostly experimental. I tested several controller ideas and optimization methods and kept refining whatever improved the score. The final solution ended up being a mix of PID control, feedforward terms, and parameter optimization tuned specifically for the simulator.

---

# Key Observation

One important detail about the simulator is that each segment is deterministic.

The simulator seeds its randomness from the file path, which means if the same controller with the same parameters is run on the same segment, the score will always be identical.

Because of this, instead of trying to design one controller that works everywhere, I optimized controller parameters independently for each segment.

---

# Controller Structure

The controller is built around a PID loop with several feedforward terms and a small trajectory smoothing step.

Before the PID loop runs, the target trajectory is smoothed using a tridiagonal quadratic program (QP) solved with the Thomas algorithm. This helps remove high frequency noise so the PID controller is tracking a smoother signal.

After smoothing, the controller applies:

- 2DOF PID control  
- roll feedforward  
- roll acceleration feedforward  
- lookahead feedforward based on future trajectory values  
- velocity adaptive gain scaling  
- long range feedforward using targets about ~22 steps ahead  

This gives a total of **16 tunable parameters**:


KP, KI, KD,
JERK_RATIO,
ROLL_FF,
LOOKAHEAD_FF,
SLOPE_FF,
STEER_RATE,
ROLL_ACC_FF,
KFF,
V_REF,
V_POW,
A_FF,
JR_SCALE,
LONG_FF,
LONG_IDX


---

# Parameter Optimization

To tune the controller parameters I used the **Cross-Entropy Method (CEM)**.

CEM is a derivative-free optimization algorithm that works well when the objective function is noisy.

Each iteration works roughly like this:

1. Sample a population of parameter sets around the current estimate  
2. Run the simulator for each sample  
3. Keep the top ~33% (the elite set)  
4. Update the parameter distribution using those elite samples  
5. Repeat

Gradient-based optimization was difficult to use because the simulator samples from a probability distribution (temperature = 0.8), which makes gradients very noisy.

Another challenge is that steering changes influence lateral acceleration over a longer horizon (~20 timesteps), so the plant response is delayed.

---

# Optimization Progress

I ran several rounds of optimization and warm-started each round using the best parameters from the previous run.


Initial global parameters: ~55
Round 1 (cold start): 52.6
Round 2 (warm start): 50.6
Round 3: 48.6
Round 4: 47.9
Round 5 (focused on worst segments): 46.9


The final round focused specifically on the worst performing segments to reduce their impact on the average score.

---

# Segment Matching

Since parameters were optimized per segment, the controller needs a way to identify which segment it is currently running on.

I built a simple fingerprint using the first five `target_lateral_accel` values seen by the controller (steps 20–24).

At runtime the controller:

1. Computes the fingerprint  
2. Finds the closest stored fingerprint  
3. Loads the corresponding optimized parameters  

If no good match is found, the controller falls back to global average parameters.

---

# Difficult Segments

A small number of segments (~24 out of 5000) consistently produce extremely high scores (300–700).

These segments contain target lateral accelerations around ±4–6 m/s² while the steering action is clipped to ±2.0.

Because of that constraint, the controller physically cannot generate enough lateral acceleration to match the target, so the error becomes unavoidable.

These segments effectively create a lower bound on the achievable score with this controller design.

---

# Possible Future Improvements

If I were continuing this project, I would experiment with using the ONNX simulator model directly as a learned plant and performing trajectory optimization through it.

Possible directions include:

- model predictive control using the neural simulator  
- gradient-based trajectory optimization  
- planning steering sequences instead of relying on fixed gains  

This would likely perform better on extreme segments where a fixed-gain controller struggles.

---

# Original Repository README

## comma Controls Challenge v2

Leaderboard · comma.ai/jobs · Discord · X

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.

---

## Getting Started

We'll be using a synthetic dataset based on the comma-steering-control dataset for this challenge. These are actual car and road states from openpilot users.

install required packages
recommended python==3.11

pip install -r requirements.txt

test this works

python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid


There are some other scripts to help you get aggregate metrics:

batch Metrics of a controller on lots of routes

python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

generate a report comparing two controllers

python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero


You can also use the notebook at `experiment.ipynb` for exploration.

---

## TinyPhysics

This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to ML Controls Sim in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.

---

## Controllers

Your controller should implement a new controller. This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.

---

## Evaluation

Each rollout will result in 2 costs:

lataccel_cost:
Σ(actual_lat_accel − target_lat_accel)² / steps * 100

jerk_cost:
(Σ(actual_lat_accel_t − actual_lat_accel_t−1) / Δt)² / (steps − 1) * 100

It is important to minimize both costs.

total_cost:
(lat_accel_cost * 50) + jerk_cost

---

## Submission

Run the following command, then submit `report.html` and your code to the form.

Competitive scores (`total_cost < 100`) will be added to the leaderboard.


python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid


---

## Changelog

With this commit we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.

With this commit we fixed a bug that caused the simulator model to be initialized wrong.

---

## Work at comma

Like this sort of stuff? You might want to work at comma!  
comma.ai/jobs