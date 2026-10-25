"""Shared exact instance format and schedule audit for the DFJSP baselines.

Times are integer ticks internally. ``time_scale`` ticks equal one input time
unit. Decimal input is scaled exactly, never rounded to a different instance.
Machine/factory IDs are zero based; distance nodes are source, M machines, sink.
"""
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def parse_dfjsp_lines(lines, name="instance"):
    lines = [line.strip().lstrip("\ufeff") for line in lines if line.strip()]
    if not lines or len(lines[0].split()) not in (4, 5):
        raise ValueError(f"{name}: expected J M F shuttle_period [average_options]")
    header = lines[0].split()
    j, m, f = map(int, header[:3])
    if min(j, m, f) <= 0:
        raise ValueError(f"{name}: positive J/M/F required")
    if len(lines) != j + m + 4:
        raise ValueError(f"{name}: expected J jobs, machine mapping and (M+2)^2 transport matrix")

    times = []

    def time_value(token, positive=False):
        try:
            value = Decimal(token)
        except InvalidOperation as exc:
            raise ValueError(f"{name}: invalid time {token}") from exc
        if not value.is_finite() or value < 0 or (positive and value == 0):
            raise ValueError(f"{name}: invalid time {token}")
        times.append(value)
        return value

    period = time_value(header[3], positive=True)
    jobs = []
    for job_index in range(j):
        tokens = lines[job_index + 1].split()
        count, cursor = int(tokens[0]), 1
        if count <= 0:
            raise ValueError(f"{name}: each job must contain operations")
        operations = []
        for _ in range(count):
            if cursor >= len(tokens):
                raise ValueError(f"{name}: truncated job {job_index}")
            alternatives = int(tokens[cursor]); cursor += 1
            if not 1 <= alternatives <= m or cursor + 2 * alternatives > len(tokens):
                raise ValueError(f"{name}: invalid/truncated alternatives")
            operation = {}
            for _ in range(alternatives):
                machine = int(tokens[cursor]) - 1
                duration = time_value(tokens[cursor + 1], positive=True)
                cursor += 2
                if machine in operation or not 0 <= machine < m:
                    raise ValueError(f"{name}: duplicate or invalid machine")
                operation[machine] = duration
            operations.append(operation)
        if cursor != len(tokens):
            raise ValueError(f"{name}: trailing tokens in job {job_index}")
        jobs.append(operations)
    mapping = list(map(int, lines[j + 1].split()))
    if len(mapping) != m or any(x < 0 or x >= f for x in mapping):
        raise ValueError(f"{name}: factory mapping must have M IDs in [0,F)")
    distance = []
    for line in lines[j + 2:]:
        tokens = line.split()
        if len(tokens) != m + 2:
            raise ValueError(f"{name}: transport matrix must be (M+2)-square")
        distance.append([time_value(token) for token in tokens])
    if any(distance[k][k] != 0 for k in range(m + 2)):
        raise ValueError(f"{name}: self-transport time must be zero")
    digits = max(max(0, -x.normalize().as_tuple().exponent) for x in times)
    if digits > 9:
        raise ValueError(f"{name}: at most 9 decimal places are supported; no rounding is applied")
    scale = 10 ** digits
    return dict(name=str(name), num_jobs=j, num_mas=m, num_fac=f,
                shuttle_period=int(period * scale), time_scale=scale,
                jobs=[[{a: int(p * scale) for a, p in op.items()} for op in job] for job in jobs],
                mac_fac_mapping=mapping,
                distance_matrix=[[int(x * scale) for x in row] for row in distance])


def parse_dfjsp_instance(filepath):
    path = Path(filepath)
    return parse_dfjsp_lines(path.read_text(encoding="utf-8-sig").splitlines(), path.name)


def departure_time(instance, dispatch, previous_machine, machine):
    factory = instance["mac_fac_mapping"]
    if previous_machine >= 0 and factory[previous_machine] != factory[machine]:
        period = instance["shuttle_period"]
        return ((dispatch + period - 1) // period) * period
    return dispatch


def collect_instance_files(data_dir):
    path = Path(data_dir)
    if path.is_file():
        return [str(path)] if path.suffix.lower() == ".fjs" else []
    return sorted(str(p) for p in path.glob("*.fjs") if p.is_file())


def write_schedule_csv(path, schedule, time_scale=1):
    """Export a checked schedule in input time units, with zero-based IDs.

    Both baselines use the same columns. In-memory schedules retain integer
    ticks; conversion happens only at this reporting boundary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["job", "op_idx", "machine", "dispatch_time", "departure_time", "start", "finish"]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=keys)
        writer.writeheader()
        for row in sorted(schedule, key=lambda item: (item["dispatch_time"], item["job"], item["op_idx"])):
            writer.writerow({key: row[key] if key in keys[:3] else row[key]/time_scale for key in keys})
    return path


def validate_schedule(instance, schedule, makespan=None, enforce_non_delay=True):
    """Independent exact audit, including coverage of every ready waiting window.

    If an operation waits in [predecessor_finish, dispatch), EVERY eligible
    machine must be continuously reserved in that window under non-delay
    dispatch. This is equivalent to the environment's lack of a WAIT action.
    ``schedule`` and optional ``makespan`` use integer ticks, not input units.
    """
    jobs, m = instance["jobs"], instance["num_mas"]
    records = {}
    by_machine = [[] for _ in range(m)]
    for item in schedule:
        key = (item["job"], item["op_idx"])
        if key in records:
            raise ValueError(f"Duplicate operation {key}")
        if not (0 <= key[0] < len(jobs) and 0 <= key[1] < len(jobs[key[0]])):
            raise ValueError(f"Invalid operation {key}")
        machine = item["machine"]
        if machine not in jobs[key[0]][key[1]]:
            raise ValueError(f"Ineligible machine for {key}")
        for field in ("dispatch_time", "departure_time", "start", "finish"):
            if not isinstance(item[field], int) or item[field] < 0:
                raise ValueError(f"{key}: {field} must be nonnegative integer ticks")
        records[key] = item
        by_machine[machine].append((item["dispatch_time"], item["finish"]))
    if len(records) != sum(map(len, jobs)):
        raise ValueError("Schedule is missing operations")
    for intervals in by_machine:
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Overlapping machine reservations")
    delivery = []
    distance = instance["distance_matrix"]
    for j, job in enumerate(jobs):
        ready, previous_machine = 0, -1
        for o, operation in enumerate(job):
            item = records[j, o]
            machine, dispatch = item["machine"], item["dispatch_time"]
            depart = departure_time(instance, dispatch, previous_machine, machine)
            travel = distance[previous_machine + 1][machine + 1]
            start = depart + travel
            finish = start + operation[machine]
            if dispatch < ready or (depart, start, finish) != (
                    item["departure_time"], item["start"], item["finish"]):
                raise ValueError(f"Invalid precedence, timetable or processing for {(j, o)}")
            for key, value in (("wait_time", depart-dispatch), ("proc_time", operation[machine]),
                               ("transport_time", travel), ("effective_time", finish-dispatch)):
                if key in item and item[key] != value:
                    raise ValueError(f"Inconsistent {key} for {(j, o)}")
            if enforce_non_delay and ready < dispatch:
                for eligible in operation:
                    covered = ready
                    for begin, end in by_machine[eligible]:
                        if end <= covered:
                            continue
                        if begin > covered or covered >= dispatch:
                            break
                        covered = end
                    if covered < dispatch:
                        raise ValueError(f"Voluntary idle time before {(j, o)} on machine {eligible}")
            ready, previous_machine = finish, machine
        delivery.append(ready + distance[previous_machine + 1][m + 1])
    result = max(delivery)
    if makespan is not None and result != makespan:
        raise ValueError("Incorrect delivery makespan")
    return result
