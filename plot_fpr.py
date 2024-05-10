import matplotlib.pyplot as plt
import numpy as np
import json

def calculate_fpr(fp, tn):
  return fp / (fp+tn)

def read_json(file_name):
    try:
        with open(file_name, 'r') as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
    except FileNotFoundError:
        print(f"File '{f}' does not exist.")
    except IOError as e:
        print(f"Error reading file: {e}")
    return content

def plot_top_k(accusation_values, epsilon, color_map, n):
    if epsilon == "e1":
        e = 1
    else:
        e = 10
    pool_values = [100000, 500000, 1000000]
    total_simulation_fprs = {}
    for pool_size in pool_values:
        total_simulation_fprs[str(pool_size)] = []
    for num_accusation in accusation_values:
        # convert naming 
        if num_accusation == 9:
            name = "0.009"
        elif num_accusation < 1000:
            name = str(num_accusation * 0.001)
        else:
            name = str(int(num_accusation*0.001))

        # read simulation results
        try:
            metrics = read_json("metrics_"+epsilon+"/metrics_"+epsilon+"_"+name+"K_accusations.json")
        except:
            print("metrics_"+epsilon+"/metrics_"+epsilon+"_"+name+"K_accusations.json does not exist.")

        for pool_size, simulations in metrics.items():
            simulation_fprs_at_n = {}
            for sim_id, (tp, tn, fp, fn) in simulations:
                if num_accusation not in simulation_fprs_at_n:
                    simulation_fprs_at_n[num_accusation] = []
                if sim_id == n:
                    fpr = calculate_fpr(fp, tn)
                    simulation_fprs_at_n[num_accusation].append(fpr)
            total_simulation_fprs[pool_size].append(simulation_fprs_at_n)

    simulations = list(accusation_values)

    # Calculate averages and variances for each simulation
    pool_avg_fpr = {}
    pool_error_margins = {}
    for key in pool_values:
        concatenated_dict = {}
        key = str(key)
        for d in total_simulation_fprs[key]:
            concatenated_dict.update(d)
        total_simulation_fprs[key] = concatenated_dict

    for key in pool_values:
        key = str(key)
        pool_avg_fpr[key] = []
        pool_error_margins[key] = []
        avg_fprs = []
        error_margins = []
        
        for num_accusation in simulations:    
            avg_fpr = np.mean(total_simulation_fprs[key][num_accusation])
            variance = np.var(total_simulation_fprs[key][num_accusation])
            avg_fprs.append(avg_fpr)
            error_margins.append(np.sqrt(variance))
        pool_avg_fpr[key] = avg_fprs
        pool_error_margins[key] = error_margins
    legends = ["100K", "500K", "1M"]
    for idx, key in enumerate(pool_values):
        key = str(key)
        color = color_map(idx/len(pool_values))
        label = rf'pool size={legends[idx]} ($\epsilon$={e})'
        plt.plot(simulations, pool_avg_fpr[key], label=label, color=color)
        plt.fill_between(simulations, np.array(pool_avg_fpr[key]) - np.array(pool_error_margins[key]), np.array(pool_avg_fpr[key]) + np.array(pool_error_margins[key]), color='gray', alpha=0.5)

    # print(pool_error_margins)
    # print(pool_avg_fpr)



n = 20 # number of colluding buyers
accusation_values = [1,2,3,4,5,6,7,8,9, 10, 20,30,40,50,60,70,80,90, 100,200,300,400,500,600,700,800,900, 1000,2000,3000,4000,5000,6000,7000,8000,9000, 10000]
simulations = list(accusation_values)

random_guess_fpr_1M = []
random_guess_fpr_500K = []
random_guess_fpr_100K = []

for simulation in simulations:
    random_guess_fpr_1M.append(simulation / 1000000)
    random_guess_fpr_100K.append(simulation / 100000)
    random_guess_fpr_500K.append(simulation / 500000)

plt.plot(simulations, random_guess_fpr_100K, linestyle='--', label = 'pool size=100K (random guess)', color="navy")
plt.plot(simulations, random_guess_fpr_500K, linestyle='--', label = 'pool size=500K (random guess)', color="blue")
plt.plot(simulations, random_guess_fpr_1M, linestyle='--', label = 'pool size=1M (random guess)', color="lightblue")
plot_top_k(accusation_values, "e1", plt.colormaps['summer'], n)
plot_top_k(accusation_values, "e10", plt.colormaps['spring'], n)

plt.xlabel('Number of Accusations', fontsize=12)
plt.ylabel('FPR', fontsize=12)

plt.legend(fontsize=8)
plt.xscale('log')
plt.ylim(0, 0.001)

y_ticks = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
plt.yticks(y_ticks, ['0.0000', '0.0001', '0.0002','0.0003','0.0004','0.0005','0.0006','0.0007','0.0008','0.0009','0.0010',])
# plt.savefig("FPR_n"+str(n)+"_top_k_extended.pdf")
# plt.show()
plt.savefig("test.pdf")