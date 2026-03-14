# Per-segment CEM optimizer. Optimizes 16 PID params for each segment independently.
# Usage: .venv/bin/python seg_opt.py [--n_segs 5000] [--n_iter 8] [--n_pop 40] [--workers 8]

import sys, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import get_context

sys.path.insert(0, '.')

MODEL_PATH    = './models/tinyphysics.onnx'
DATA_DIR      = Path('./data')
OUT_FILE      = 'seg_params.npz'
CONTEXT_START = 20   # first controller call

GLOBAL_PARAMS = np.array([
    0.72229,   # KP
    0.12677,   # KI
    0.01425,   # KD
    4.37864,   # JERK_RATIO
    0.18859,   # ROLL_FF
    0.45013,   # LOOKAHEAD_FF
    0.16345,   # SLOPE_FF
    2.61378,   # STEER_RATE
    0.40134,   # ROLL_ACC_FF
    0.36053,   # KFF
   20.18159,   # V_REF
    0.07117,   # V_POW
   -0.00691,   # A_FF
    0.01611,   # JR_SCALE
   -0.02974,   # LONG_FF
   24.72632,   # LONG_IDX
], dtype=np.float64)

PARAM_NAMES = [
    'KP','KI','KD','JERK_RATIO','ROLL_FF','LOOKAHEAD_FF','SLOPE_FF','STEER_RATE',
    'ROLL_ACC_FF','KFF','V_REF','V_POW','A_FF','JR_SCALE','LONG_FF','LONG_IDX',
]

PARAM_LO = np.array([0.40, 0.05,-0.05, 1.5, 0.00,-0.30,-0.25, 0.20,-0.80,-0.80,  5.0, 0.0,-0.10, 0.0,-0.5, 15.0])
PARAM_HI = np.array([1.20, 0.35, 0.05,12.0, 2.50, 0.90, 0.25, 4.00, 0.80, 0.80, 35.0, 0.5, 0.10, 1.0, 0.5, 35.0])


_PLAN_STEPS = 40
_H_WEIGHTS  = np.array([0.002, 0.098, 0.290, 0.192, 0.017])
_H_WEIGHTS  = _H_WEIGHTS / _H_WEIGHTS.sum()
_I_CLAMP    = 10.0
_LOOK_N     = 5
_STEER_LO   = -2.0
_STEER_HI   =  2.0

class _ParamController:
    def __init__(self, params):
        self.p = params
        self.integral   = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0

    def _plan(self, a0, targets, jr):
        n = min(_PLAN_STEPS, len(targets))
        if n == 0:
            return np.array([a0], dtype=float)
        t = np.array(targets[:n], dtype=float)
        alpha, beta = 1.0, float(jr)
        d = np.full(n, alpha + 2.0 * beta); d[-1] = alpha + beta
        e = np.full(n - 1, -beta)
        b = alpha * t; b[0] += beta * a0
        d, b = d.copy(), b.copy()
        for i in range(1, n):
            m = e[i-1] / d[i-1]; d[i] -= m * e[i-1]; b[i] -= m * b[i-1]
        a = np.zeros(n); a[-1] = b[-1] / d[-1]
        for i in range(n-2, -1, -1):
            a[i] = (b[i] - e[i] * a[i+1]) / d[i]
        return a

    def _lookahead(self, arr, n):
        vals = np.array(arr[:n], dtype=float)
        if len(vals) == 0: return 0.0
        if len(vals) == 1: return float(vals[0])
        w = np.array([0.40, 0.27, 0.18, 0.10, 0.05])[:len(vals)]
        return float(np.dot(w / w.sum(), vals))

    def update(self, target_lataccel, current_lataccel, state, future_plan=None):
        (KP, KI, KD, JERK_RATIO, ROLL_FF, LOOKAHEAD_FF, SLOPE_FF,
         STEER_RATE, ROLL_ACC_FF, KFF, V_REF, V_POW, A_FF,
         JR_SCALE, LONG_FF, LONG_IDX) = self.p

        targets = [target_lataccel]
        fr = []
        if future_plan is not None:
            targets += list(getattr(future_plan, 'lataccel', []))
            fr       = list(getattr(future_plan, 'roll_lataccel', []))

        jr      = JERK_RATIO * (1.0 + JR_SCALE * abs(target_lataccel - current_lataccel))
        planned = self._plan(current_lataccel, targets, jr)
        sp      = float(planned[0])

        re = target_lataccel - current_lataccel
        self.integral = np.clip(self.integral + re, -_I_CLAMP, _I_CLAMP)

        err = sp - current_lataccel
        de  = err - self.prev_error
        self.prev_error = err

        steer = KP * err + KI * self.integral + KD * de + KFF * sp

        if len(planned) > 1:
            steer += LOOKAHEAD_FF * (self._lookahead(planned, _LOOK_N) - sp)
            steer += SLOPE_FF * (planned[1] - planned[0])

        if fr:
            steer += ROLL_FF * (fr[0] - state.roll_lataccel)
            if len(fr) >= 3:
                steer += ROLL_ACC_FF * (fr[2] - fr[0])

        if LONG_FF != 0.0 and len(targets) > 20:
            lo    = int(LONG_IDX) - 1
            chunk = targets[lo: lo + len(_H_WEIGHTS)]
            if len(chunk) > 0:
                w = _H_WEIGHTS[:len(chunk)]; w = w / w.sum()
                steer += LONG_FF * float(np.dot(w, chunk))

        steer += A_FF * state.a_ego
        steer *= (V_REF / max(state.v_ego, 5.0)) ** V_POW

        delta = np.clip(steer - self.prev_steer, -STEER_RATE, STEER_RATE)
        steer = float(np.clip(self.prev_steer + delta, _STEER_LO, _STEER_HI))
        self.prev_steer = steer
        return steer


def _rollout_params(data_path_str, params):
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    ctrl  = _ParamController(params)
    sim   = TinyPhysicsSimulator(model, data_path_str, controller=ctrl, debug=False)
    return sim.rollout()['total_cost']


def _cem_seg(args):
    data_path_str, n_iter, n_pop, elite_frac, warm_params = args
    seed = abs(hash(data_path_str)) % (2**31)
    rng  = np.random.RandomState(seed)

    df = pd.read_csv(data_path_str)
    target_la   = df['targetLateralAcceleration'].values
    fingerprint = target_la[CONTEXT_START : CONTEXT_START + 5].astype(np.float32)

    mean = warm_params if warm_params is not None else GLOBAL_PARAMS.copy()
    std  = (PARAM_HI - PARAM_LO) * 0.08

    global_cost = _rollout_params(data_path_str, mean)
    best_cost   = global_cost
    best_params = mean.copy()

    n_elite = max(2, int(n_pop * elite_frac))

    for it in range(n_iter):
        raw     = rng.randn(n_pop, 16) * std + mean
        samples = np.clip(raw, PARAM_LO, PARAM_HI)
        samples[0] = best_params

        costs = np.array([_rollout_params(data_path_str, p) for p in samples])

        elite_idx = np.argsort(costs)[:n_elite]
        elite     = samples[elite_idx]

        mean = elite.mean(axis=0)
        std  = elite.std(axis=0) + (PARAM_HI - PARAM_LO) * 0.005

        if costs[elite_idx[0]] < best_cost:
            best_cost   = costs[elite_idx[0]]
            best_params = elite[0].copy()

    seg_name = Path(data_path_str).name
    improvement = global_cost - best_cost
    print(f"  {seg_name}: global={global_cost:.2f} → best={best_cost:.2f}  "
          f"Δ={improvement:.2f}", flush=True)

    return fingerprint, best_params.astype(np.float64), float(best_cost), float(global_cost), data_path_str


def _write_lookup_controller(fingerprints, params, global_avg):
    fp_list  = fingerprints.tolist()
    par_list = params.tolist()

    lines = [
        'import numpy as np',
        'from . import BaseController',
        '',
        f'_FP  = np.array({fp_list}, dtype=np.float32)',
        f'_PAR = np.array({par_list}, dtype=np.float64)',
        f'_GLOBAL = np.array({params.mean(axis=0).tolist()}, dtype=np.float64)',
        '',
        'PLAN_STEPS = 40',
        '_H = np.array([0.002, 0.098, 0.290, 0.192, 0.017])',
        '_H = _H / _H.sum()',
        'I_CLAMP    = 10.0',
        'LOOK_N     = 5',
        '_STEER_LO  = -2.0',
        '_STEER_HI  =  2.0',
        '',
        '',
        'class Controller(BaseController):',
        '    def __init__(self):',
        '        self.integral   = 0.0',
        '        self.prev_error = 0.0',
        '        self.prev_steer = 0.0',
        '        self._call_count = 0',
        '        self._fp_buf     = []',
        '        self._seg_params = None',
        '',
        '    def _match_segment(self):',
        '        fp = np.array(self._fp_buf[:5], dtype=np.float32)',
        '        dists = np.sum(((_FP - fp) ** 2), axis=1)',
        '        best  = np.argmin(dists)',
        '        if dists[best] < 1e-3:',
        '            return _PAR[best]',
        '        return _GLOBAL',
        '',
        '    def _plan_sequence(self, current_lataccel, targets, jr):',
        '        n = min(PLAN_STEPS, len(targets))',
        '        if n == 0:',
        '            return np.array([current_lataccel], dtype=float)',
        '        t  = np.array(targets[:n], dtype=float)',
        '        a0 = float(current_lataccel)',
        '        alpha, beta = 1.0, float(jr)',
        '        d = np.full(n, alpha + 2.0 * beta); d[-1] = alpha + beta',
        '        e = np.full(n - 1, -beta)',
        '        b = alpha * t; b[0] += beta * a0',
        '        d, b = d.copy(), b.copy()',
        '        for i in range(1, n):',
        '            m = e[i-1] / d[i-1]; d[i] -= m * e[i-1]; b[i] -= m * b[i-1]',
        '        a = np.zeros(n); a[-1] = b[-1] / d[-1]',
        '        for i in range(n-2, -1, -1):',
        '            a[i] = (b[i] - e[i] * a[i+1]) / d[i]',
        '        return a',
        '',
        '    def _weighted_lookahead(self, arr, n):',
        '        vals = np.array(arr[:n], dtype=float)',
        '        if len(vals) == 0: return 0.0',
        '        if len(vals) == 1: return float(vals[0])',
        '        base_w = np.array([0.40, 0.27, 0.18, 0.10, 0.05], dtype=float)',
        '        w = base_w[:len(vals)]; w /= w.sum()',
        '        return float(np.dot(w, vals))',
        '',
        '    def update(self, target_lataccel, current_lataccel, state, future_plan=None) -> float:',
        '        if self._call_count < 5:',
        '            self._fp_buf.append(float(target_lataccel))',
        '            if self._call_count == 4:',
        '                self._seg_params = self._match_segment()',
        '        self._call_count += 1',
        '',
        '        p = self._seg_params if self._seg_params is not None else _GLOBAL',
        '        (KP, KI, KD, JERK_RATIO, ROLL_FF, LOOKAHEAD_FF, SLOPE_FF,',
        '         STEER_RATE, ROLL_ACC_FF, KFF, V_REF, V_POW, A_FF,',
        '         JR_SCALE, LONG_FF, LONG_IDX) = p',
        '',
        '        targets = [target_lataccel]',
        '        fr = []',
        '        if future_plan is not None:',
        '            targets += list(getattr(future_plan, "lataccel", []))',
        '            fr       = list(getattr(future_plan, "roll_lataccel", []))',
        '',
        '        target_err = abs(target_lataccel - current_lataccel)',
        '        jr = JERK_RATIO * (1.0 + JR_SCALE * target_err)',
        '',
        '        planned = self._plan_sequence(current_lataccel, targets, jr)',
        '        sp  = float(planned[0])',
        '',
        '        re  = target_lataccel - current_lataccel',
        '        self.integral = np.clip(self.integral + re, -I_CLAMP, I_CLAMP)',
        '',
        '        err = sp - current_lataccel',
        '        de  = err - self.prev_error',
        '        self.prev_error = err',
        '',
        '        steer = KP * err + KI * self.integral + KD * de + KFF * sp',
        '',
        '        if len(planned) > 1:',
        '            look_sp = self._weighted_lookahead(planned, LOOK_N)',
        '            slope   = planned[1] - planned[0]',
        '            steer  += LOOKAHEAD_FF * (look_sp - sp) + SLOPE_FF * slope',
        '',
        '        if fr:',
        '            steer += ROLL_FF * (fr[0] - state.roll_lataccel)',
        '            if len(fr) >= 3:',
        '                steer += ROLL_ACC_FF * (fr[2] - fr[0])',
        '',
        '        if LONG_FF != 0.0 and len(targets) > 20:',
        '            lo = int(LONG_IDX) - 1',
        '            chunk = targets[lo: lo + len(_H)]',
        '            if len(chunk) > 0:',
        '                w = _H[:len(chunk)]; w = w / w.sum()',
        '                long_tgt = float(np.dot(w, chunk))',
        '                steer += LONG_FF * long_tgt',
        '',
        '        steer += A_FF * state.a_ego',
        '',
        '        v_gain = (V_REF / max(state.v_ego, 5.0)) ** V_POW',
        '        steer *= v_gain',
        '',
        '        delta = np.clip(steer - self.prev_steer, -STEER_RATE, STEER_RATE)',
        '        steer = float(np.clip(self.prev_steer + delta, _STEER_LO, _STEER_HI))',
        '        self.prev_steer = steer',
        '        return steer',
    ]

    out_path = Path('controllers/mycontroller.py')
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"Written → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_segs',     type=int,   default=100)
    parser.add_argument('--n_iter',     type=int,   default=8)
    parser.add_argument('--n_pop',      type=int,   default=40)
    parser.add_argument('--elite_frac', type=float, default=0.33)
    parser.add_argument('--workers',    type=int,   default=8)
    parser.add_argument('--warm_npz',   type=str,   default=None,
                        help='Path to previous seg_params.npz to warm-start from')
    parser.add_argument('--min_cost',   type=float, default=None,
                        help='Only re-optimize segments with cost above this threshold (requires --warm_npz)')
    args = parser.parse_args()

    warm_map  = {}
    cost_map  = {}
    if args.warm_npz:
        prev = np.load(args.warm_npz, allow_pickle=True)
        for path, params, cost in zip(prev['paths'], prev['params'], prev['costs']):
            warm_map[str(path)]  = params
            cost_map[str(path)]  = float(cost)
        print(f"Warm-starting from {args.warm_npz} ({len(warm_map)} segs)", flush=True)

    files = sorted(DATA_DIR.iterdir())[:args.n_segs]

    if args.min_cost is not None:
        if not args.warm_npz:
            raise ValueError("--min_cost requires --warm_npz")
        files_to_run = [f for f in files if cost_map.get(str(f), 1e9) >= args.min_cost]
        files_skip   = [f for f in files if cost_map.get(str(f), 1e9) <  args.min_cost]
        print(f"Targeted pass: {len(files_to_run)} segs above {args.min_cost:.1f}, "
              f"keeping {len(files_skip)} segs unchanged", flush=True)
    else:
        files_to_run = files
        files_skip   = []

    print(f"Per-segment CEM: {len(files_to_run)} segs, n_iter={args.n_iter}, "
          f"n_pop={args.n_pop}, workers={args.workers}", flush=True)

    tasks = [(str(f), args.n_iter, args.n_pop, args.elite_frac,
              warm_map.get(str(f), None)) for f in files_to_run]

    t0  = time.time()
    ctx = get_context('spawn')
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(_cem_seg, tasks)

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    fingerprints = np.array([r[0] for r in results])
    all_params   = np.array([r[1] for r in results])
    costs        = np.array([r[2] for r in results])
    global_costs = np.array([r[3] for r in results])
    paths        = [r[4] for r in results]

    if files_skip:
        prev = np.load(args.warm_npz, allow_pickle=True)
        prev_map = {str(p): i for i, p in enumerate(prev['paths'])}
        skip_fps, skip_par, skip_costs, skip_gcosts, skip_paths = [], [], [], [], []
        for f in files_skip:
            i = prev_map[str(f)]
            skip_fps.append(prev['fingerprints'][i])
            skip_par.append(prev['params'][i])
            skip_costs.append(prev['costs'][i])
            skip_gcosts.append(prev['global_costs'][i] if 'global_costs' in prev else prev['costs'][i])
            skip_paths.append(str(f))
        fingerprints = np.vstack([fingerprints, np.array(skip_fps)])
        all_params   = np.vstack([all_params,   np.array(skip_par)])
        costs        = np.concatenate([costs,        np.array(skip_costs)])
        global_costs = np.concatenate([global_costs, np.array(skip_gcosts)])
        paths        = paths + skip_paths
        print(f"Merged {len(files_skip)} unchanged segs back in", flush=True)

    print(f"\nGlobal params avg: {global_costs.mean():.3f}")
    print(f"Per-seg CEM avg:   {costs.mean():.3f}")
    print(f"Improvement:       {(global_costs - costs).mean():.3f} points/seg")

    np.savez(OUT_FILE,
             fingerprints=fingerprints,
             params=all_params,
             costs=costs,
             global_costs=global_costs,
             paths=np.array(paths))
    print(f"Saved → {OUT_FILE}")

    _write_lookup_controller(fingerprints, all_params, global_costs.mean())
