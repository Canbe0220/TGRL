"""Read the original DFJSP format (including source/sink transport times)."""
import math
import torch


def nums_detec(lines):
    if not lines or len(lines[0].split()) < 4:
        raise ValueError('Expected header: jobs machines factories shuttle_period [flexibility]')
    header = lines[0].split()
    jobs, machines, factories = map(int, header[:3])
    period = float(header[3])
    if min(jobs, machines, factories) <= 0 or not math.isfinite(period) or period <= 0:
        raise ValueError('Positive dimensions and finite positive shuttle period are required')
    if len(lines) < jobs + 1:
        raise ValueError('Missing job records')
    counts = [int(lines[i].split()[0]) if lines[i].split() else 0 for i in range(1, jobs + 1)]
    if min(counts) <= 0:
        raise ValueError('Each job must contain at least one operation')
    return jobs, machines, factories, period, sum(counts)


def load_fjs(lines, num_mas, num_opes, device='cpu'):
    jobs, machines, factories, _, total = nums_detec(lines)
    if machines != num_mas or total > num_opes:
        raise ValueError('Incorrect padded instance dimensions')
    proc = torch.zeros((num_opes, num_mas), dtype=torch.float32, device=device)
    pre = torch.zeros((num_opes, num_opes), dtype=torch.bool, device=device)
    cumul = torch.zeros((num_opes, num_opes), dtype=torch.int64, device=device)
    owner = torch.zeros(num_opes, dtype=torch.long, device=device)
    starts, counts, offset = [], [], 0
    for j in range(jobs):
        tokens = lines[j + 1].split()
        count, k = int(tokens[0]), 1
        starts.append(offset); counts.append(count)
        for local in range(count):
            if k >= len(tokens):
                raise ValueError('Truncated operation record')
            n_options = int(tokens[k]); k += 1
            if not 1 <= n_options <= num_mas:
                raise ValueError('Invalid number of eligible machines')
            seen = set()
            for _ in range(n_options):
                if k + 1 >= len(tokens):
                    raise ValueError('Truncated processing-time record')
                m, duration = int(tokens[k]) - 1, float(tokens[k + 1]); k += 2
                if m in seen or not 0 <= m < num_mas or not math.isfinite(duration) or duration <= 0:
                    raise ValueError('Invalid machine or processing time')
                seen.add(m); proc[offset + local, m] = duration
            o = offset + local
            owner[o] = j
            if local:
                pre[o - 1, o] = True
                cumul[offset:o, o] = 1
        if k != len(tokens):
            raise ValueError('Unexpected trailing tokens in job record')
        offset += count
    extra = [line.split() for line in lines[jobs + 1:] if line.strip()]
    if len(extra) != num_mas + 3:
        raise ValueError('Expected machine-factory mapping and (M+2)-square transport matrix')
    mapping = torch.tensor(list(map(int, extra[0])), dtype=torch.long, device=device)
    if mapping.numel() != num_mas or mapping.min() < 0 or mapping.max() >= factories:
        raise ValueError('Factory IDs must be in [0, number of factories)')
    if any(len(row) != num_mas + 2 for row in extra[1:]):
        raise ValueError('Transport matrix must be (M+2)-square')
    distance = torch.tensor([[float(v) for v in row] for row in extra[1:]], dtype=torch.float32, device=device)
    if not torch.isfinite(distance).all() or (distance < 0).any():
        raise ValueError('Transport times must be finite and nonnegative')
    if not torch.allclose(distance.diagonal(), torch.zeros(num_mas + 2, device=device)):
        raise ValueError('Self-transport time must be zero')
    return (proc, (proc > 0).long(), pre, pre.t(), owner,
            torch.tensor(starts, dtype=torch.long, device=device),
            torch.tensor(counts, dtype=torch.long, device=device), cumul, mapping, distance)
