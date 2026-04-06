# HyCAPS
HyCAPS (Hybrid CMA-PSO) is a two-islands optimization method which leverages Covariance Matrix Adaptation-Evolution Strategy and Fuzzy Self-Tuning Particle Swarm Optimization.

## Example

FST-PSO can be used as follows:

	from fstpso import FuzzyPSO	
	from hycaps import HyCAPS

	def example_fitness( particle ):
		return sum(map(lambda x: x**2, particle))
		
	
	if __name__ == "__main__":
	    sea = HyCAPS()
	    sea.set_search_space([[-30, 30]] * 2)
	    sea.set_cma_strarting_matrix()
	    sea.set_fitness(Ackley)
	    solution, fitness = sea.solve_with_hycaps()
	    print(solution, fitness)
	    sea.dump_best_fitnesses("test.txt")

## Further information

FST-PSO has been created by Daniele M. Papetti, Daniela Besozzi (University of Milano-Bicocca),
Marco S. Nobile and Matteo Grazioso (Ca' Foscari University of Venice), Paolo Cazzaniga 
(University of Bergamo), and Leonardo Vanneschi (Universidade NOVA de Lisboa). 
The source code is written and maintained by D.M. Papetti.

If you need any information about FST-PSO please write to: daniele.papetti@unimib.it

HyCAPS requires fst-pso and deap.

The work has been presented at EvoStar 2026 (publication in-print).