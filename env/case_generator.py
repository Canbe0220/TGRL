import random
import time
import numpy as np

class CaseGenerator:
    '''
    FJSP instance generator
    '''
    def __init__(self, job_init, num_mas, num_fac, opes_per_job_min, opes_per_job_max, nums_ope=None, path='data_dev/100302/',
                 flag_same_opes=True, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas
        self.num_fac = num_fac

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = self.num_mas
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = 20
        self.proctime_dev = 0.2

        self.trans_intra_min = 0
        self.trans_intra_max = 10
        self.trans_inter_min = 10
        self.trans_inter_max = 20

        self.shuttle_period_min = 10
        self.shuttle_period_max = 20


    def get_case(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas + 1), val))
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)]
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope

        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        shuttle_period = random.randint(self.shuttle_period_min, self.shuttle_period_max)

        line0 = '{0} {1} {2} {3} {4}\n'.format(self.num_jobs, self.num_mas, self.num_fac, shuttle_period, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0} {1} {2} {3} {4}'.format(self.num_jobs, self.num_mas, self.num_fac, shuttle_period, self.num_options / self.num_opes))

        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        
        mas_fac_assignment = sorted([m % self.num_fac for m in range(self.num_mas)])
        fac_mapping_line = " ".join(map(str, mas_fac_assignment))
        
        lines.append(fac_mapping_line + '\n')
        lines_doc.append(fac_mapping_line)

        new_total_nodes = self.num_mas + 2
        distance_matrix = [[0] * new_total_nodes for _ in range(new_total_nodes)]
        for i in range(1, self.num_mas + 1):
            for j in range(i + 1, self.num_mas + 1):
                if mas_fac_assignment[i - 1] == mas_fac_assignment[j - 1]:
                    dist = random.randint(self.trans_intra_min, self.trans_intra_max)
                else:
                    dist = random.randint(self.trans_inter_min, self.trans_inter_max)
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        for i in range(1, self.num_mas + 1):
            dist_start = random.randint(self.trans_intra_min, self.trans_intra_max)
            distance_matrix[0][i] = dist_start
            distance_matrix[i][0] = dist_start
        
            dist_end = random.randint(self.trans_intra_min, self.trans_intra_max)
            distance_matrix[i][new_total_nodes - 1] = dist_end
            distance_matrix[new_total_nodes - 1][i] = dist_end
        
        distance_matrix[0][new_total_nodes - 1] = 0
        distance_matrix[new_total_nodes - 1][0] = 0
            
        
        for row in distance_matrix:
            matrix_row_str = " ".join(map(str, row))
            lines.append(matrix_row_str + '\n')
            lines_doc.append(matrix_row_str)

        lines.append('\n')
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}f_{3}.fjs'.format(self.num_jobs, self.num_mas, self.num_fac, str(idx + 1).zfill(3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_mas
