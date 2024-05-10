import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import uuid
import mmh3
import hashlib
from bitarray import bitarray
import json
import time
import os

l_1 = 65536

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size  # Number of bits in the Bloom Filter
        self.num_hashes = num_hashes  # Number of hash functions
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hash(self, uid, hash_function):
        # Create a hash object using the specified hash function
        hasher = hashlib.new(hash_function)
        hasher.update(uid.encode('utf-8'))
        return int(hasher.hexdigest(), 16) % self.size

    
    def add(self, uid):
        index = []
        for i in range(self.num_hashes):
            # seed = generate_seed(uid, i)
            # Create a new hash for each of the m hash functions
            hash_result = mmh3.hash(uid, i) % self.size
            # Set the bit at the hash index to 1
            self.bit_array[hash_result] = 1
            index.append(hash_result)
        #assert(len(index)==self.num_hashes)
        return index

    def set(self, idx):
      for i in idx:
        self.bit_array[i] = 1

    def check(self, uid):
        for i in range(self.num_hashes):
            # use the same m hash functions
            hash_result = mmh3.hash(uid, i) % self.size
            # If any bit at the hash index is 0, then uid is definitely not in the set
            if self.bit_array[hash_result] == 0:
                return False
        return True


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

def write_json(content, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(content, json_file)
    print(f"content saved to {file_path}")

def laplace_random(b, mu=0):
    U = np.random.uniform()
    return mu - b * np.sign(U - 0.5) * np.log(1 - 2 * abs(U - 0.5))

def laplace_pdf(x, b, mu=0):
    return (1 / (2 * b)) * np.exp(-abs(x-mu) / b)

def generate_unique_uids(m):
    unique_uids = set()
    uid_length = 5  # should cover around 1M UIDs
    while len(unique_uids) < m:
        uid = str(uuid.uuid4())[:uid_length]
        unique_uids.add(uid)
    return list(unique_uids)

def simulate_visit(full_user_list, num_user_visit):
    user_visited = random.sample(full_user_list, num_user_visit)
    return set(user_visited)

def report(user_visited, user_hash, n, kv_pair, num_hashes):
    indices = []
    for uid in user_visited:
      truth = user_hash[uid]
      for idx in truth:
        indices.append(idx)
        kv_pair[idx] += (l_1 / num_hashes) * n
    return kv_pair
    #print(len(indices)-len(set(indices)))


def query(kv_pair, num_bits, epsilon):
    noised_output = []
    for bucket in range(num_bits):
       noised_output.append(kv_pair[bucket] + laplace_random(l_1/epsilon, 0))
    return noised_output # vector of m

def generate_user_score(noisy_vector, target_user_list, user_hash, n, epsilon, num_hashes):
    user_score = {}
    for uid in target_user_list:
       target_index = user_hash[uid]
       score = 1
       for idx in target_index:
          noised_output= noisy_vector[idx]
          pdf_c = laplace_pdf(noised_output - (l_1 / num_hashes) * n, l_1/epsilon, 0)
          pdf_0 = laplace_pdf(noised_output, l_1/epsilon, 0)
          prob_h1 = pdf_c / (pdf_c + pdf_0)
          score *= prob_h1
       user_score[uid] = score
    return user_score


def predict(user_score, num_accusations):
    sorted_scores = sorted(user_score.items(), key=lambda item: item[1], reverse=True)
    predict_visited = [item[0] for item in sorted_scores[:num_accusations]]
    return predict_visited

def calculate_metrics(pool_size, num_user_visit, user_visited, predict_visited):
    tp = len(user_visited.intersection(predict_visited))
    tn = pool_size - len(user_visited.union(predict_visited))
    fp = len(predict_visited) - tp
    fn = num_user_visit - tp
    return tp, tn, fp, fn

def calculate_ppv(tp, tn, fp, fn):
  if tp == 0:
    return 0
  else:
    return tp / (tp + fp)

def calculate_fpr(tp, tn, fp, fn):
  return fp / (fp + tn)


def generate_1M_UIDs():
    start_time = time.time()
    candidate_uids = generate_unique_uids(1000000)
    file_path = "1M_UIDs.json"
    with open(file_path, "w") as file:
        json.dump(candidate_uids, file)
    print(f"UIDs saved to {file_path}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

def generate_metrics_for_high_PPV(user_visited, scores, num_accusation): #PPV >= 0.99
    ppvs = []
    flag = 0
    for user_score in scores:
        predict_visited = predict(user_score, num_accusation)
        current_tp, current_tn, current_fp, current_fn = calculate_metrics(1000000, 10000, set(user_visited), set(predict_visited))
        current_ppv = calculate_ppv(current_tp, current_tn, current_fp, current_fn)
        ppvs.append(current_ppv)
    averaged_ppv = np.mean(ppvs)
    variance = np.var(ppvs)
    if(averaged_ppv >= 0.99):
        flag = 1
        return flag, averaged_ppv, variance
    else:
        return flag, averaged_ppv, variance
    
# assume 10K users actually visit out of 1M candidate pool
def find_high_ppv(candidates, epsilon, n, num_accusations):
    count = 0
    num_simulation = 1
    num_bits = 201000
    num_hashes = 20
    inter_state = {}
    metrics = {} #(epsilon:..., n:..., num_accusation:..., avg_ppv:..., variance:...)
    bloom_filter = BloomFilter(num_bits, num_hashes)
    user_hash = {}
    for user in candidates:  
        hash = bloom_filter.add(user)
        user_hash[user] = hash 
    user_visited = simulate_visit(candidates, num_user_visit=10000)
    inter_state["UID_visited"] = list(user_visited)
    inter_state["scores"] = []
    kv_pair = np.zeros(num_bits)
    kv_pair = report(user_visited, user_hash, n, kv_pair, num_hashes)
    for _ in range(num_simulation):
        noisy_vector = query(kv_pair, num_bits, epsilon)
        user_score = generate_user_score(noisy_vector, candidates, user_hash, n, epsilon, num_hashes)  
        inter_state["scores"].append(user_score)
    for num_accusation in num_accusations: # num_accusations need to rank from small to large
        flag, averaged_ppv, variance = generate_metrics_for_high_PPV(user_visited, inter_state["scores"], num_accusation)
        print(flag)
        # print(averaged_ppv)
        # print(variance)
        if flag == 1:
                metrics["epsilon"] = epsilon 
                metrics["colluding_buyers"] = n 
                metrics["num_accusation"] = num_accusation 
                metrics["avg_ppv"] = averaged_ppv
                metrics["var_ppv"] = variance
                name = "_e"+str(epsilon)+"_n_"+str(n)+"_top_"+str(num_accusation)
                interstate_file_path = "interstate/interstate"+name+".json"
                metrics_file_path = "metrics/metrics"+name+".json"
                write_json(inter_state, interstate_file_path)
                write_json(metrics, metrics_file_path)
                count += 1
                return flag
                
    return flag
    

def calculate_num_colluder_metrics():
    for filename in os.listdir("num_colluders/"):
        filepath = os.path.join("num_colluders/", filename)
        num = read_json(filepath)
        mean = np.mean(num)
        variance = np.var(num)
        print(filename+" mean: "+str(mean)+" var: "+str(variance))
        
epsilon = 10
num_accusations = [1000, 5000, 10000]
candidates = read_json("1M_UIDs.json")

for num_accusation in num_accusations:
    num_colluders = []
    # estimate and adjust n_values based on epsilon and num_accusation
    if num_accusation == 1000:
        n_values = range(1, 10)
    if num_accusation == 5000:
        n_values = range(3, 15)
    if num_accusation == 10000:
        n_values = range(15, 35)
    for i in range(5):
        for n in n_values:
            flag = find_high_ppv(candidates, epsilon , n, [num_accusation])
            print((n, flag))
            if flag == 1:
                num_colluders.append(n)
                break
    write_json(num_colluders, "num_colluders/e"+str(epsilon)+"_top_"+str(num_accusation)+".json")

# after finish all simulations
calculate_num_colluder_metrics()