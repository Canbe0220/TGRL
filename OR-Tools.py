"""OR-Tools CP-SAT for heterogeneous-transport reservation DFJSP.
"""
import argparse
import csv
import math
from pathlib import Path
from time import perf_counter

from ortools.sat.python import cp_model
from dfjsp_common import (collect_instance_files, parse_dfjsp_instance,
                         validate_schedule, write_schedule_csv)
from PDRs import RULES, solve_pdr_schedule

DATA_DIRS = ["data_test/100502"]

TIME_LIMIT = 1800
NUM_WORKERS = 8
LOG_SEARCH = False
STOP_AFTER_FIRST_FEASIBLE = False
PRINT_IMPROVEMENTS = True
PDR_RULES_FOR_HINT = RULES
ENFORCE_NON_DELAY = True
SAVE_DIR = "optimal"


def solve_pdr_reservation(instance, rule):
    """Warm start from precisely the four PDRs evaluated in PDRs.py."""
    return solve_pdr_schedule(instance, rule, check=True)


def best_pdr_hint(instance):
    results = [(obj, schedule, rule) for rule in PDR_RULES_FOR_HINT
               for obj, schedule in [solve_pdr_reservation(instance, rule)]]
    return min(results, key=lambda result: result[0])


def build_cp_model(instance, upper_bound, hint, enforce_non_delay=True):
    jobs, m = instance["jobs"], instance["num_mas"]
    factory, distance = instance["mac_fac_mapping"], instance["distance_matrix"]
    period, horizon = instance["shuttle_period"], int(upper_bound)
    if horizon < 1 or horizon + period >= 2**60:
        raise ValueError("Scaled horizon is outside the supported CP-SAT integer range")
    model = cp_model.CpModel()
    operations, intervals, compatible = {}, [[] for _ in range(m)], [[] for _ in range(m)]
    for j, job in enumerate(jobs):
        for o, choices in enumerate(job):
            key = j, o
            label = f"j{j}_o{o}"
            variables = {name: model.new_int_var(0, horizon, f"{name}_{label}")
                         for name in ("dispatch", "depart", "start", "finish", "duration")}
            slot = model.new_int_var(0, (horizon+period-1)//period, f"slot_{label}")
            variables["slot"] = slot
            # ceil(dispatch / P), including boarding exactly at a departure.
            model.add(slot * period >= variables["dispatch"])
            model.add(slot * period <= variables["dispatch"] + period - 1)
            model.add(variables["duration"] == variables["finish"] - variables["dispatch"])
            variables["choice"] = {}
            for machine, processing in choices.items():
                selected = model.new_bool_var(f"choose_{label}_m{machine}")
                variables["choice"][machine] = selected
                intervals[machine].append(model.new_optional_interval_var(
                    variables["dispatch"], variables["duration"], variables["finish"],
                    selected, f"reservation_{label}_m{machine}"))
                compatible[machine].append(key)
                model.add(variables["finish"] == variables["start"] + processing).only_enforce_if(selected)
            model.add_exactly_one(list(variables["choice"].values()))
            operations[key] = variables
    for (j, o), variables in operations.items():
        d, dep, start = variables["dispatch"], variables["depart"], variables["start"]
        if o == 0:
            model.add(dep == d)  # source transport has immediate departure
            for machine, selected in variables["choice"].items():
                model.add(start == dep + distance[0][machine+1]).only_enforce_if(selected)
        else:
            previous = operations[j, o-1]
            model.add(d >= previous["finish"])
            for source, was_selected in previous["choice"].items():
                for target, selected in variables["choice"].items():
                    active = [was_selected, selected]
                    model.add(start == dep + distance[source+1][target+1]).only_enforce_if(active)
                    if factory[source] == factory[target]:
                        model.add(dep == d).only_enforce_if(active)
                    else:
                        model.add(dep == variables["slot"] * period).only_enforce_if(active)
    for machine in range(m):
        model.add_no_overlap(intervals[machine])
        if enforce_non_delay and compatible[machine]:
            # B_m(t) is 0/1 occupancy, W_m(t) counts ready, not-dispatched
            # operations compatible with m. 0 <= J*B_m(t)-W_m(t) <= J forces
            # W_m(t)=0 whenever m is idle. At most one operation per job waits,
            # so W_m<=J; the constraint imposes nothing extra when m is busy.
            # One reservoir per machine avoids per-time or per-wait-window grids.
            times, changes, actives = [], [], []
            for j, o in compatible[machine]:
                v = operations[j, o]
                ready = 0 if o == 0 else operations[j, o-1]["finish"]
                selected = v["choice"][machine]
                times.extend((v["dispatch"], v["finish"], ready, v["dispatch"]))
                changes.extend((len(jobs), -len(jobs), -1, 1))
                actives.extend((selected, selected, True, True))
            model.add_reservoir_constraint_with_active(times, changes, actives, 0, len(jobs))
    deliveries = []
    for j, job in enumerate(jobs):
        last = operations[j, len(job)-1]
        delivery = model.new_int_var(0, horizon, f"delivery_j{j}")
        deliveries.append(delivery)
        for machine, selected in last["choice"].items():
            model.add(delivery == last["finish"] + distance[machine+1][m+1]).only_enforce_if(selected)
    objective = model.new_int_var(0, horizon, "delivery_makespan")
    model.add_max_equality(objective, deliveries)
    model.minimize(objective)
    by_operation = {(row["job"], row["op_idx"]): row for row in hint}
    for key, row in by_operation.items():
        v = operations[key]
        values = dict(dispatch=row["dispatch_time"], depart=row["departure_time"],
                      start=row["start"], finish=row["finish"],
                      duration=row["finish"]-row["dispatch_time"],
                      slot=(row["dispatch_time"]+period-1)//period)
        for name, value in values.items():
            model.add_hint(v[name], value)
        for machine, selected in v["choice"].items():
            model.add_hint(selected, int(machine == row["machine"]))
    for j, job in enumerate(jobs):
        row = by_operation[j, len(job)-1]
        model.add_hint(deliveries[j], row["finish"] + distance[row["machine"]+1][m+1])
    model.add_hint(objective, horizon)
    return model, operations, objective, sum(map(len, intervals))


class ImprovementPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, objective, scale, prefix=""):
        super().__init__()
        self.objective, self.scale, self.prefix = objective, scale, prefix
        self.best = math.inf

    def on_solution_callback(self):
        value = self.value(self.objective) / self.scale
        if value < self.best:
            self.best = value
            print(f"{self.prefix} feasible upper bound={value:g} solver_time={self.wall_time:.2f}s", flush=True)


def solve_dfjsp_cp_sat(instance, time_limit=TIME_LIMIT, num_workers=NUM_WORKERS,
                            log_search=False, prefix="", enforce_non_delay=ENFORCE_NON_DELAY,
                            seed=1, print_improvements=None):
    """Solve reservation DFJSP with CP-SAT under the requested dispatch scope.
    """
    if not math.isfinite(time_limit) or time_limit <= 0 or num_workers < 1:
        raise ValueError("Positive finite time limit and worker count required")
    started = perf_counter()
    scale = instance.get("time_scale", 1)
    pdr_ub, pdr_schedule, pdr_rule = best_pdr_hint(instance)
    pdr_seconds = perf_counter() - started
    model, variables, objective, n_intervals = build_cp_model(
        instance, pdr_ub, pdr_schedule, enforce_non_delay)
    error = model.validate()
    if error:
        raise ValueError(f"Invalid CP-SAT model: {error}")
    build_seconds = perf_counter() - started - pdr_seconds
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search)
    solver.parameters.stop_after_first_solution = STOP_AFTER_FIRST_FEASIBLE
    solver.parameters.random_seed = int(seed)
    display = PRINT_IMPROVEMENTS if print_improvements is None else print_improvements
    callback = ImprovementPrinter(objective, scale, prefix) if display else None
    status = solver.solve(model, callback)
    status_name = solver.status_name(status)
    if status in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
        # A validated PDR schedule witnesses feasibility. Never hide a model bug
        # as an apparently successful PDR fallback.
        raise RuntimeError(f"CP-SAT returned {status_name} despite a feasible warm start: {solver.response_stats()}")
    fallback = status not in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if fallback:
        schedule, upper = pdr_schedule, pdr_ub
    else:
        schedule = []
        upper = solver.value(objective)
        for (j, o), v in variables.items():
            machine = next(m for m, chosen in v["choice"].items() if solver.boolean_value(chosen))
            schedule.append(dict(job=j, op_idx=o, machine=machine,
                                 dispatch_time=solver.value(v["dispatch"]),
                                 departure_time=solver.value(v["depart"]),
                                 start=solver.value(v["start"]), finish=solver.value(v["finish"])))
    validate_schedule(instance, schedule, upper, enforce_non_delay)
    lower = max(0.0, solver.best_objective_bound) / scale
    if lower > upper / scale + 1e-7:
        raise RuntimeError("Inconsistent CP-SAT objective bound")
    return dict(status=status_name + ("_PDR_FALLBACK" if fallback else ""),
                proven_optimal=status == cp_model.OPTIMAL,
                objective_upper_bound=upper / scale, solver_lower_bound=lower,
                relative_gap=(upper/scale-lower)/max(abs(upper/scale), 1e-12),
                runtime=solver.wall_time, total_runtime=perf_counter()-started,
                pdr_runtime=pdr_seconds, build_runtime=build_seconds,
                horizon=pdr_ub / scale, time_scale=scale, num_intervals=n_intervals,
                pdr_upper_bound=pdr_ub / scale, pdr_rule=pdr_rule, used_fallback=fallback,
                dispatch_mode="non_delay" if enforce_non_delay else "allow_idle",
                schedule=schedule)


# Historical name retained; default scope is the non-delay scheduling domain.
solve_dfjsp_full_cp_sat = solve_dfjsp_cp_sat


def solve_folder(data_dir, output_dir=SAVE_DIR, time_limit=TIME_LIMIT, num_workers=NUM_WORKERS,
                 enforce_non_delay=ENFORCE_NON_DELAY, limit=None, seed=1,
                 log_search=False, print_improvements=None):
    files = collect_instance_files(data_dir)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .fjs instances in {data_dir}")
    folder = Path(data_dir).stem if Path(data_dir).is_file() else Path(data_dir).name
    output = Path(output_dir)
    if not enforce_non_delay:
        output = output / "allow_idle"
    output.mkdir(parents=True, exist_ok=True)
    schedule_dir = output / (folder + "_schedules"); schedule_dir.mkdir(exist_ok=True)
    values = []
    columns = ["folder", "instance_index", "instance_name", "status", "proven_optimal",
               "objective_upper_bound", "solver_lower_bound", "relative_gap", "runtime",
               "total_runtime", "pdr_runtime", "build_runtime", "horizon", "time_scale",
               "num_intervals", "pdr_upper_bound", "pdr_rule", "used_fallback", "dispatch_mode"]
    with (output / f"{folder}.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=columns); writer.writeheader()
        for index, path in enumerate(files, 1):
            instance = parse_dfjsp_instance(path)
            prefix = f"[{folder}] {index}/{len(files)} {Path(path).name}"
            result = solve_dfjsp_cp_sat(instance, time_limit, num_workers, log_search, prefix,
                                            enforce_non_delay, seed, print_improvements)
            schedule = result.pop("schedule")
            writer.writerow(dict(folder=folder, instance_index=index, instance_name=Path(path).name, **result))
            target.flush(); values.append(result["objective_upper_bound"])
            write_schedule_csv(schedule_dir / (Path(path).stem + ".csv"), schedule, result["time_scale"])
            print(f"{prefix}: {result['status']}, UB={result['objective_upper_bound']:g}, "
                  f"LB={result['solver_lower_bound']:g}, mode={result['dispatch_mode']}", flush=True)
        mean = sum(values) / len(values)
        writer.writerow(dict(folder=folder, instance_index="AVG", instance_name="average_upper_bound",
                             objective_upper_bound=mean, dispatch_mode="non_delay" if enforce_non_delay else "allow_idle"))
    print(f"Saved {output / (folder + '.csv')}; mean feasible makespan={mean:g}", flush=True)
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", default=DATA_DIRS)
    parser.add_argument("--output-dir", default=SAVE_DIR)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-idle", action="store_true", help="Broader physical scheduling domain; NOT TGRL's action set")
    parser.add_argument("--log-search", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable incumbent progress callbacks")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for directory in args.data_dirs:
        solve_folder(directory, args.output_dir, args.time_limit, args.workers, not args.allow_idle,
                     args.limit, args.seed, args.log_search, not args.quiet)


if __name__ == "__main__":
    main()
