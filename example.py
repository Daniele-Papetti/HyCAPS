import math
from hycaps import HyCAPS


def Ackley(particle):
    n = len(particle)
    accum = 20.0 + math.e
    sub1 = 0.0
    for i in range(n):
        sub1 = sub1 + math.pow(particle[i], 2)
    sub1 = sub1/n
    accum = accum - 20.0*math.exp(-0.2*math.sqrt(sub1))
    sub2 = 0.0
    for i in range(n):
        sub2 = sub2 + math.cos(2*particle[i]*math.pi)
    sub2 = sub2/n
    accum = accum - math.exp(sub2)
    return accum


if __name__ == "__main__":
    sea = HyCAPS()
    sea.set_search_space([[-30, 30]] * 2)
    sea.set_cma_strarting_matrix()
    sea.set_fitness(Ackley)
    solution, fitness = sea.solve_with_hycaps()
    print(solution, fitness)
    sea.dump_best_fitnesses("test.txt")
