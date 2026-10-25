import numpy as np
import random
import torch
from env.case_generator import CaseGenerator

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Generate instances and save to files
def main():
    setup_seed(100)
    batch_size = 100
    num_jobs = 20
    num_mas = 10
    num_fac = 3
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
    nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
    case = CaseGenerator(num_jobs, num_mas, num_fac, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope, path='data_test/201003/', flag_same_opes=True, flag_doc=True)
    for i in range(batch_size):
        case.get_case(i)

if __name__ == "__main__":
    main()
    print("Data_dev is created. ")