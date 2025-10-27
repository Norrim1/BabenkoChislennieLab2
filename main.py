import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, log, sin, lambdify
from scipy.optimize import minimize_scalar

def f(x):
    return (np.log(x))**2 + np.sin(x)


start = np.pi
end = 7*np.pi
x_nodes = np.linspace(start, end, 11)
y_nodes = f(x_nodes)

def lagrange_poly(x, x_nodes, y_nodes):
    total = 0
    n = len(x_nodes)
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if j != i:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        total += term
    return total


def practical_error(x, x_nodes, y_nodes):
    return abs(f(x) - lagrange_poly(x, x_nodes, y_nodes))


def theoretical_error_bound(x, x_nodes):
    n = len(x_nodes) - 1
    a = min(x_nodes)
    b = max(x_nodes)

    t = symbols('t')
    f_symbol_form = (log(t))**2 + sin(t)
    derivative_symbol_form = diff(f_symbol_form, t, n + 1)
    f_derivative = lambdify(t, derivative_symbol_form, modules=['numpy']) 

    def minus_abs(x):
        return -abs(f_derivative(x))
    
    res = minimize_scalar(minus_abs, bounds=(a, b), method='bounded')
    max_x_on_graph = -minus_abs(res.x)

    return (max_x_on_graph * np.prod([abs(x - xi) for xi in x_nodes])) / math.factorial(n + 1)


x_vals = np.linspace(start, end, 100)
y_vals = f(x_vals)
lagrange_vals = [lagrange_poly(x, x_nodes, y_nodes) for x in x_vals]

practical_errs = [practical_error(x, x_nodes, y_nodes) for x in x_vals]
theoretical_errs = [theoretical_error_bound(x, x_nodes) for x in x_vals]

plt.figure(figsize=(10,6))
plt.plot(x_vals, y_vals, label="Сама функция")
plt.plot(x_nodes, y_nodes, 'o', label="Узлы")
plt.plot(x_vals, lagrange_vals, '--', label="Лагранжа")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(x_vals, practical_errs, label="Практическая ошибка")
plt.plot(x_vals, theoretical_errs, '--', label="Теоретическая ошибка")
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Ошибка')
plt.legend()
plt.grid(True)
plt.show()
