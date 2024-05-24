import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
from scipy.special import comb
from mpmath import mp

n_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# l_1 = 2**16
k_values = [2, 100, 1000, 10000, 100000, 1000000] #number of candidate users
epsilon = 1
color_map = plt.get_cmap('summer')

def midpoint_rule_approximation_positive(L, m, k):
  delta_x = L / m
  total_sum = 0

  for j in range(1, L+1):
    x_j = (2 * j - 1) * delta_x / 2
    f_x_j = 0.5 * np.exp(-epsilon * x_j) * (1-0.5*np.exp(-epsilon * (n+x_j)))**(k-1)
    total_sum += f_x_j

  return total_sum * delta_x

def midpoint_rule_approximation_negative(n, m, k):
    delta_x = n / m
    total_sum = 0

    for j in range(1, m + 1):
        x_j = -n + (2 * j - 1) * delta_x / 2
        f_x_j = 0.5 * epsilon * np.exp(-epsilon * abs(x_j)) * (1 - 0.5 * np.exp(-epsilon * (n + x_j)))**(k-1)
        total_sum += f_x_j

    return total_sum * delta_x

def midpoint_rule_approximation(L, m, k):
  delta_x = L / m
  total_sum = 0

  for j in range(1, m+1):
    x_j = (2 * j - 1) * delta_x / 2
    f_x_j = 0.5 * np.exp(-epsilon * x_j) * (1-0.5*np.exp(-epsilon * (n+x_j)))**(k-1)
    total_sum += f_x_j

  return total_sum * delta_x

# alternative: scipy integral approximation 
def laplace_integral(n, k):
    # n + delta_j < 0
    def integrand1(delta_j):
        return 0.5 * epsilon * np.exp(-epsilon * abs(delta_j)) * (0.5 * np.exp(epsilon * (n + delta_j)))**(k-1)

    # n + delta_j > 0
    def integrand2(delta_j):
        return 0.5 * epsilon * np.exp(-epsilon * abs(delta_j)) * (1 - 0.5 * np.exp(-epsilon * (n + delta_j)))**(k-1)

    # Evaluate the two integrals
    integral1, _ = quad(integrand1, -np.inf, -n)
    integral2, _ = quad(integrand2, -n, 0)
    integral3, _ = quad(integrand2, 0, np.inf)

    # Return the sum of the two integrals
    return integral1 + integral2 + integral3

def expression(n, k):
  #return (1-0.5 * np.exp(-n))**(k-1) + (1/(2*k*2**k))*np.exp(-n)
  return 0.5 * (1-0.5 * np.exp(-n * epsilon))**(k-1)

def expression_below_negative_n(n, k):
  # return (epsilon / (k * 2**k)) * np.exp(-n * epsilon)
  return (1 / (k * 2**k)) * np.exp(-n * epsilon)


k_values = [2, 100, 1000, 10000, 100000, 1000000] #number of candidate users
labels = ["2", "100", "1K", "10K", "100K", "1M"]
for idx, k in enumerate(k_values):
  accuracies = []
  for n in n_values:
    # accuracies.append(laplace_integral(n, k)) #directly approximate integrals with build-in functions
    accuracies.append(midpoint_rule_approximation_positive(1000, 10000, k)+midpoint_rule_approximation_negative(n, n*10, k)+expression_below_negative_n(n, k))
  # print(accuracies)
  label = f'k={labels[idx]}'
  color = color_map(idx/len(k_values))
  plt.plot(n_values, accuracies, label=label, color=color)

#plt.title(f'Expected Accuracy of Prediction when epsilon={epsilon}')
plt.xlabel('Number of Colluding Buyers', fontsize=12)
# plt.ylabel('P(x_j > x_i for all i not equal to j in k)', fontsize=12)
plt.ylabel(r'P($x_j$ is the Largest among k Outputs)', fontsize=12)
plt.legend(fontsize=10)

plt.xticks(n_values, n_values)
y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.yticks(y_ticks)

plt.savefig('Accuracy.pdf', dpi=300, bbox_inches='tight')