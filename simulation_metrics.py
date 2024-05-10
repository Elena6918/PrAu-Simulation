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

def calculate_metrics(user_visited, predict_visited):
    tp = len(user_visited.intersection(predict_visited))
    tn = len(candidates) - len(user_visited.union(predict_visited))
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

####################################################################
# Define constants
color_map = plt.colormaps['summer']
pool_values = [100000, 500000, 1000000]
num_simulation = 5
num_bits = 201000
num_hashes = 20
num_user_visit = 10000  # number of insertions, assume 10K user visits
epsilon = 10 
n_values = range(1, 21) # adjust according to epsilon
# n_values = range(5, 125, 5)
num_accusations = 10000 # adjust according to need

start_time = time.time()

candidate_uids = generate_unique_uids()

inter_state = {}
inter_state["UID_visited"] = []
tps, tns, fps, fns = [], [], [], []
metrics = {} # save metrics of simulation results

for idx, pool in enumerate(pool_values):
  tp = Counter()
  tn = Counter()
  fp = Counter()
  fn = Counter()
  candidates = generate_unique_uids(pool)
  bloom_filter = BloomFilter(num_bits, num_hashes)
  user_hash = {}
  collision = 0
  for user in candidates:
    hash = bloom_filter.add(user)
    user_hash[user] = hash 
  user_visited = simulate_visit(candidates)

  metrics[pool] = []
  inter_state[pool] = []
  for n in n_values:
      print(n)
      for _ in range(num_simulation):
          kv_pair = np.zeros(num_bits)
          report(user_visited, user_hash, n)
          noisy_vector = query(kv_pair)
          user_score = generate_user_score(noisy_vector, candidates, user_hash, n)  

          predict_visited = predict(user_score, num_accusations)
          current_tp, current_tn, current_fp, current_fn = calculate_metrics(user_visited, set(predict_visited))
          tp[n] += current_tp
          tn[n] += current_tn
          fp[n] += current_fp
          fn[n] += current_fn
          metrics[pool].append((n, [current_tp, current_tn, current_fp, current_fn]))
          inter_state[pool].append((n, user_score))
          inter_state["UID_visited"].append((pool, list(user_visited)))
  
  name = str(pool)

file_path = "metrics_e10_10K_accusations.json"
with open(file_path, "w") as file:
    json.dump(metrics, file)
print(f"Metrics saved to {file_path}")


file_path = "interstate_e10.json"
with open(file_path, "w") as file:
    json.dump(inter_state, file)
print(f"Interstate saved to {file_path}")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")
