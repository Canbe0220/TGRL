"""Four timetable-aware PDRs for the SAME reservation DFJSP as TGRL.

No torch/OR-Tools dependency. Times in instance dictionaries and schedules are
exact integer ticks; solve_by_pdr() returns makespan in original input units.
CSV metrics and per-rule schedule files use original input units. The -T
suffix denotes the transport-aware adaptation of each classical priority rule.
"""
import argparse
import csv
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from dfjsp_common import (collect_instance_files, departure_time,
                         parse_dfjsp_instance, validate_schedule, write_schedule_csv)

DATA_DIRS = [f"data_test/{key}" for key in (
    "100502", "200502", "151003",
    "201003", "301003", "401003",
    "Public_02")]
RULES = ["FIFO-T", "SPT-T", "MOR-T", "MWKR-T"]
PDR_OUTPUT_DIR = "PDRs"


def build_transport_estimator(instance):
    """FIFO earliest-arrival DP, including timetables and final delivery.
    """
    jobs, distance = instance["jobs"], instance["distance_matrix"]
    sink = instance["num_mas"] + 1

    @lru_cache(maxsize=16384)
    def route_finish(job, operation, source, ready):
        frontier = {source: ready}
        for choices in jobs[job][operation:]:
            frontier = {
                target: min(departure_time(instance, finish, previous, target)
                            + distance[previous + 1][target + 1] + processing
                            for previous, finish in frontier.items())
                for target, processing in choices.items()
            }
        return min(finish + distance[machine + 1][sink]
                   for machine, finish in frontier.items())
    return route_finish


def select_by_rule(candidates, rule):
    rule = {x[:-2]: x for x in RULES}.get(rule, rule)
    # Integer IDs are deterministic tie-breaks, not numerical scheduling scores.
    if rule == "FIFO-T":
        key = lambda c: (c["job_ready"], c["job"], c["candidate_route_cost"],
                         c["effective_time"], c["machine"])
    elif rule == "SPT-T":
        key = lambda c: (c["effective_time"], c["candidate_route_cost"],
                         c["job_ready"], c["job"], c["machine"])
    elif rule == "MOR-T":
        key = lambda c: (-c["remaining_ops"], c["candidate_route_cost"],
                         c["effective_time"], c["job"], c["machine"])
    elif rule == "MWKR-T":
        key = lambda c: (-c["transport_remaining_work"], c["candidate_route_cost"],
                         c["effective_time"], c["job"], c["machine"])
    else:
        raise ValueError(f"Unknown PDR rule: {rule}")
    return min(candidates, key=key)


def solve_pdr_schedule(instance, rule, check=True):
    """Return (delivery makespan in ticks, complete schedule in ticks).

    FIFO-T: earliest-ready job, then earliest relaxed delivered finish.
    SPT-T: shortest current reservation (shuttle wait + travel + processing).
    MOR-T: most remaining operations, then earliest relaxed delivered finish.
    MWKR-T: largest relaxed remaining duration including shuttle wait/delivery,
            then earliest relaxed delivered finish for machine assignment.
    Job-level workload is identical across that job's machine alternatives.
    """
    if rule not in RULES and rule not in [r[:-2] for r in RULES]:
        raise ValueError(f"Unknown PDR rule: {rule}")
    jobs = instance["jobs"]
    j, m = instance["num_jobs"], instance["num_mas"]
    distance = instance["distance_matrix"]
    route_finish = build_transport_estimator(instance)
    steps, ready, last_machine = [0] * j, [0] * j, [-1] * j
    available, delivered = [0] * m, [0] * j
    now, schedule, total = 0, [], sum(map(len, jobs))
    while len(schedule) < total:
        candidates = []
        for job in range(j):
            operation = steps[job]
            if operation == len(jobs[job]) or ready[job] > now:
                continue
            source = last_machine[job]
            workload = route_finish(job, operation, source, now) - now
            for machine, processing in jobs[job][operation].items():
                if available[machine] > now:
                    continue
                depart = departure_time(instance, now, source, machine)
                travel = distance[source + 1][machine + 1]
                start, finish = depart + travel, depart + travel + processing
                candidates.append(dict(
                    job=job, op_idx=operation, machine=machine,
                    dispatch_time=now, departure_time=depart, start=start, finish=finish,
                    proc_time=processing, transport_time=travel, wait_time=depart-now,
                    effective_time=finish-now, job_ready=ready[job],
                    remaining_ops=len(jobs[job])-operation,
                    transport_remaining_work=workload,
                    candidate_route_cost=route_finish(job, operation+1, machine, finish)-now))
        if not candidates:
            future = [t for t in available if t > now]
            if not future:
                raise RuntimeError("No eligible action or future reservation completion")
            now = min(future)
            continue
        item = select_by_rule(candidates, rule)
        schedule.append(item)
        job, machine = item["job"], item["machine"]
        available[machine] = ready[job] = item["finish"]
        last_machine[job] = machine
        steps[job] += 1
        if steps[job] == len(jobs[job]):
            delivered[job] = item["finish"] + distance[machine + 1][m + 1]
    makespan = max(delivered)
    if check:
        validate_schedule(instance, schedule, makespan, enforce_non_delay=True)
    return makespan, schedule


def solve_by_pdr(instance, rule, check=True, return_schedule=False):
    """Input-unit score; with return_schedule=True, also return integer-tick rows."""
    makespan, schedule = solve_pdr_schedule(instance, rule, check)
    value = makespan / instance.get("time_scale", 1)
    return (value, schedule) if return_schedule else value


def average_makespan_for_folder(data_dir, rule):
    files = collect_instance_files(data_dir)
    return (sum(solve_by_pdr(parse_dfjsp_instance(p), rule) for p in files) / len(files)
            if files else None)


def evaluate_and_save_folder(data_dir, output_dir=PDR_OUTPUT_DIR, limit=None):
    files = collect_instance_files(data_dir)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .fjs instances in {data_dir}")
    folder = Path(data_dir).stem if Path(data_dir).is_file() else Path(data_dir).name
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    values = {r: [] for r in RULES}
    runtimes = {r: [] for r in RULES}
    with (output / f"{folder}.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=["Instance", *RULES, *[r+"_seconds" for r in RULES],
                                                   "dispatch_mode", "time_scale"])
        writer.writeheader()
        for index, path in enumerate(files, 1):
            instance = parse_dfjsp_instance(path)
            scale = instance.get("time_scale", 1)
            row = {"Instance": Path(path).name, "dispatch_mode": "non_delay", "time_scale": scale}
            for rule in RULES:
                started = perf_counter()
                makespan, schedule = solve_pdr_schedule(instance, rule)
                elapsed = perf_counter() - started
                value = makespan / scale
                row[rule], row[rule+"_seconds"] = value, elapsed
                values[rule].append(value); runtimes[rule].append(elapsed)
                write_schedule_csv(output / (folder + "_schedules") / rule / (Path(path).stem + ".csv"),
                                   schedule, scale)
            writer.writerow(row); target.flush()
            print(f"[{folder}] {index}/{len(files)} {Path(path).name}: "
                  + ", ".join(f"{r}={row[r]:g}" for r in RULES), flush=True)
        averages = {r: sum(values[r])/len(files) for r in RULES}
        writer.writerow({"Instance": "Average", "dispatch_mode": "non_delay", **averages,
                         **{r+"_seconds": sum(runtimes[r])/len(files) for r in RULES}})
    print(f"Saved {output / (folder + '.csv')}", flush=True)
    return averages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", default=DATA_DIRS)
    parser.add_argument("--output-dir", default=PDR_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="Maximum instances per directory")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for directory in args.data_dirs:
        means = evaluate_and_save_folder(directory, args.output_dir, args.limit)
        print(Path(directory).name + ": " + ", ".join(f"{r}={means[r]:.4f}" for r in RULES))


if __name__ == "__main__":
    main()
