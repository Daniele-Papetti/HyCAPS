import os
import sys
import uuid
import pickle
import logging
import numpy as np
from fstpso import FuzzyPSO
from deap import creator, cma, base


def checkBounds(lbs, ubs):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > ubs[i]:
                        child[i] = ubs[i]
                    elif child[i] < lbs[i]:
                        child[i] = lbs[i]
            return offspring
        return wrapper
    return decorator


class HyCAPS:
    def __init__(self,
                 Ncma=67,
                 Npso=33,
                 iterations=100,
                 migrations_frequency=1,
                 search_space=None,
                 centroid=None,
                 sigma=None,
                 disable_vminflag=False,
                 record_best_fitnesses=True,
                 record_migration_patterns=False):
        """
        Create a new HyCAPS instance.

        :param Ncma: Size of the CMA-ES island popultion.
        :param Npso: Size of the FST-PSO island population.
        :param iterations: Number of iterations of HyCAPS.
        :param migrations_frequency: Parameter k of HyCAPS. It represents the frequency of the best solution migrations.
        :param search_space: Search Space of the optimization problem. Search space is a list where the d-th element is
        a couple of lower and upper bound of the d-th dimensions of the search space.
        :param centroid: List representing the starting centroid for the CMA-ES island.
        :param sigma: Starting sigma for the CMA-ES island.
        :param disable_vminflag: Flag used to enable/disable the minimum rule velocity of the FST-PSO island.
        :param record_best_fitnesses: Flag used to enable/disable the recording of the best fitness found during each
        iteration of HyCAPS. With this flag enabled, it is possible to call dump_best_fitnesses() to dump the best
        fitnesses found on a text file.
        :param record_migration_patterns: Flag used to enable/disable the recording of the migrations between the
        islands. With this flag enabled, it is possible to call dump_migration_patterns() to dump the migration
        patterns on a pickle file.
        """
        self._Npso = Npso
        self._Ncma = Ncma
        self._iterations = iterations
        self._K = migrations_frequency

        self.set_search_space(search_space)
        if self._search_space is not None or (centroid is not None and sigma is not None):
            self.set_cma_strarting_matrix(centroid, sigma)
        self._fitness = None
        self._fitness_args = None
        self._vmin_flag = disable_vminflag

        self._cma = None
        self._fstpso = None
        self._fstpso_checkpointname = ""
        self._cma_bests_fit = []
        self._fstpso_bests_fit = []
        self._best_fitnesses = [] if record_best_fitnesses else None
        self._migrations = [] if record_migration_patterns else None
        self._solution_fitness = sys.float_info.max
        self._solution = None

    def solve_with_hycaps(self):
        """
        Run HyCAPS to solve the optimization problem.

        :return: A numpy-array containing the best solution found and the respective fitness value.
        """
        self._setup_cmaes()
        self._setup_fstpso()
        logging.info(f"Checkpoint name: {self._fstpso_checkpointname}")

        for it in range(self._iterations):
            # perform CMA-ES step
            population = self._toolbox.generate()
            cma_fitnesses = list(self._toolbox.map(self._toolbox.evaluate, population))
            for ind, fit in zip(population, cma_fitnesses):
                ind.fitness.values = fit

            # perform FST-PSO step
            self._fstpso._actually_launch_optimization(verbose=False)
            fstpso_fitnesses = [x.CalculatedBestFitness for x in self._fstpso.Solutions]
            self._fstpso._load_checkpoint(self._fstpso_checkpointname)

            # determine best and worst of CMA
            min_cma_idx = np.argmin(cma_fitnesses)
            max_cma_idx = np.argmax(cma_fitnesses)
            min_cma_fitness = cma_fitnesses[min_cma_idx][0]

            # determine best and worst of FSTPSO
            min_fstpso_idx = np.argmin(fstpso_fitnesses)
            max_fstpso_idx = np.argmax(fstpso_fitnesses)
            min_fstpso_fitness = self._fstpso.Solutions[min_fstpso_idx].CalculatedFitness

            self._update_islands_bests(min_cma_fitness, min_fstpso_fitness)

            # migrate
            if it % self._K == self._K - 1:
                # move best of CMA to FSTPSO swarm
                if min_cma_fitness < min_fstpso_fitness:
                    logging.info(f" * Iteration {it}, new min solution migrating from CMA-ES to FST-PSO: {min_cma_fitness}")
                    self._fstpso.Solutions[max_fstpso_idx].X = population[min_cma_idx]
                    self._fstpso.Solutions[max_fstpso_idx].V = list(np.zeros(self._D))
                    self._fstpso.Solutions[max_fstpso_idx].CalculatedFitness = min_cma_fitness
                    self._fstpso.Solutions[max_fstpso_idx].cannot_move = True
                    if self._migrations is not None:
                        self._migrations.append((min_cma_fitness, 'fstpso', it))

                # move best of FSTPSO to CMA
                elif min_fstpso_fitness < min_cma_fitness:
                    logging.info(f" * Iteration {it}, new min solution migrating from FST-PSO to CMA-ES: {min_fstpso_fitness}")
                    population[max_cma_idx] = creator.Individual(self._fstpso.Solutions[min_fstpso_idx].X[:])
                    population[max_cma_idx].fitness.values = (min_fstpso_fitness, )
                    if self._migrations is not None:
                        self._migrations.append((min_fstpso_fitness, 'cmaes', it))
            # update CMA covariance matrix
            self._toolbox.update(population)

            self._update_solution(min_cma_fitness, population[min_cma_idx],
                                  min_fstpso_fitness, self._fstpso.Solutions[min_fstpso_idx])

        os.remove(self._fstpso_checkpointname)
        return self._solution, self._solution_fitness

    def _update_islands_bests(self,
                              min_cma,
                              min_fstpso):
        """
        Update the best solution found by each island.

        :param min_cma (float): Fitness of the best solution found by the CMA-ES island during the current iteration.
        :param min_fstpso (float): Fitness of the best solution found by the FST-PSO island during
        the current iteration.
        """
        if len(self._cma_bests_fit) == 0:
            self._cma_bests_fit.append(min_cma)
            self._fstpso_bests_fit.append(min_fstpso)
            return
        if min_cma < self._cma_bests_fit[-1]:
            self._cma_bests_fit.append(min_cma)
        else:
            self._cma_bests_fit.append(self._cma_bests_fit[-1])
        if min_fstpso < self._fstpso_bests_fit[-1]:
            self._fstpso_bests_fit.append(min_fstpso)
        else:
            self._fstpso_bests_fit.append(self._fstpso_bests_fit[-1])

    def _update_solution(self,
                         best_cma_fitness,
                         best_cma_ind,
                         best_fstpso_fitness,
                         best_fstpso_particle):
        """
        Update the best solution found by HyCAPS.

        :param best_cma_fitness: Fitness of the best solution found by the CMA-ES island during the current iteration.
        :param best_cma_ind: Best solution found by the CMA-ES island during the current iteration.
        :param best_fstpso_fitness: Fitness of the best solution found by the FST-PSO island during
        the current iteration.
        :param best_fstpso_particle: Best solution found by the FST-PSO island during the current iteration.
        """
        if best_cma_fitness < self._solution_fitness:
            self._solution_fitness = best_cma_fitness
            self._solution = np.array(self._toolbox.clone(best_cma_ind))
        if best_fstpso_fitness < self._solution_fitness:
            self._solution_fitness = best_fstpso_fitness
            self._solution = np.array(best_fstpso_particle.X[:])
        if self._best_fitnesses is not None:
            self._best_fitnesses.append(self._solution_fitness)

    def _setup_cmaes(self):
        """
        Create the deap toolbox used to run the CMA-ES island.
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", self._wrap_fitness_function_for_cma)
        self._cma = cma.Strategy(centroid=self._centroid, sigma=self._sigma, lambda_=self._Ncma, mu=int(self._Ncma / 2))
        self._toolbox.register("update", self._cma.update)
        self._toolbox.register("generate", self._cma.generate, creator.Individual)
        lbs, ubs = list(zip(*[(lb, ub) for lb, ub in self._search_space]))
        self._toolbox.decorate("generate", checkBounds(lbs, ubs))

    def _setup_fstpso(self):
        """
        Create the FST-PSO instance to run the FST-PSO island.
        """
        self._fstpso = FuzzyPSO()
        self._fstpso.set_search_space(self._search_space)
        self._fstpso.set_fitness(self._call_fitness, skip_test=True)
        self._fstpso.set_swarm_size(self._Npso)
        if self._vmin_flag:
            self._fstpso.disable_fuzzyrule_minvelocity()
        self._fstpso_checkpointname = f"checkpoint_{uuid.uuid1()}"
        self._fstpso.solve_with_fstpso(max_iter=1, save_checkpoint=self._fstpso_checkpointname)

    def _call_fitness(self, x):
        """
        Compute the fitness of a candidate solution.

        :param x (ArrayLike): Coordinate of the candidate solution.
        :return: Fitness of the candidate solution.
        """
        if self._fitness is None:
            raise Exception("The fitness function was not set")
        if self._fitness_args is None:
            return self._fitness(x)
        else:
            return self._fitness_args(x, self._fitness_args)

    def _wrap_fitness_function_for_cma(self, x):
        """
        Wrapper to call the fitness function for the CMA-ES isalnd.

        :param x (ArrayLike): Coordinate of the candidate solution.
        :return: Tuple of one element containing the fitness of the candidate solution.
        """
        return self._call_fitness(x),

    def set_fitness(self,
                    fitness,
                    arguments=None):
        """
        Set the fitness function of the optimization problem to be solved.

        :param fitness: Function to compute the fitness of a candidate solution. The fitness functions must receive as
        a first argument a List containing the coordinates of a candidate solution.
        If the fitness function uses additional arguments, the fitness function has to receive such arguments stored in
        a dictionary as the second argument of the fitness function.
        :param arguments: Optional additional arguments for the fitness function.
        Such arguments must be stored in a dictionary.
        """
        self._fitness = fitness
        self._fitness_args = arguments

    def set_search_space(self,
                         search_space):
        """
        Set the search space of the optimization problem.

        :param search_space: Search Space of the optimization problem. Search space is a list where the d-th element is
        a couple of lower and upper bound of the d-th dimensions of the search space.
        """
        if search_space is None:
            self._D = 0
            self._search_space = None
        else:
            self._D = len(search_space)
            self._search_space = search_space
        logging.info(f"Search space set ({self._D} dimensions)")

    def set_cma_strarting_matrix(self,
                                 centroid=None,
                                 sigma=None):
        """
        Set the centroid and the sigma of the starting covariance matrix for the CMA-ES island.

        :param centroid: List representing the starting centroid for the CMA-ES island.
        If the centroid is passaed as None (default value) the centroid will be automatically set.
        :param sigma: Starting sigma for the CMA-ES island.
        If the sigma is passed as None (default value) the sigma will be automatically set.
        """
        if centroid is None:
            if self._search_space is not None:
                self._centroid = [0] * self._D
            else:
                self._centroid = None
                logging.error("The centroid cannot be automatically set if the search space is not defined")
        else:
            self._centroid = centroid
        if sigma is None:
            if self._search_space is not None:
                self._sigma = (self._search_space[0][1] - self._search_space[0][0]) / 6
            else:
                self._sigma = None
                logging.error("The sigma cannot be automatically set if the search space is not defined")
        else:
            self._sigma = sigma

    def dump_best_fitnesses(self,
                            dump_file):
        """
        The best solution found during each iteration of HyCAPS is saved on a text file.
        Such method can be called only if the record_best_fitnesses attribute was set to True in the
        initialization of the object.

        :param dump_file (str): Path where the file be dumped.
        """
        if self._best_fitnesses is not None:
            np.savetxt(dump_file, self._best_fitnesses)
        else:
            logging.info("The tracking of the best fitnesses was disabled during the run.\n"
                         "Set record_best_fitnesses=True in the init function to track the fitnesses")

    def dump_islands_best_fitnesses(self,
                                    dump_file,
                                    cma_postfix='_cmaes',
                                    pso_postfix='_fstpso'):
        """
        The best solutions found by each island during each iteration of HyCAPS are saved on two separate .txt files.

        :param dump_file: Path where to dump the files. Please note that such argument specifies the path and the shared
        part of the names where the files will be dumped. See examples below.
        :param cma_postfix: Last part of the name for the best solutions found by the CMA-ES island.
        :param pso_postfix: Last part of the name for the best solutions found by the FST-PSO island.
        """
        np.savetxt(f"{dump_file}{cma_postfix}.txt", self._cma_bests_fit)
        np.savetxt(f"{dump_file}{pso_postfix}.txt", self._fstpso_bests_fit)

    def dump_migration_patterns(self,
                                dump_file):
        """
        Save the migration patterns to a pickle file. Such method can be called only if the record_migration_patterns
        attribute is set to True during the initialization of the object.
        The pickle file will contain of list of tuples of three elements:
        (fitness of the migrating individual, island of destination, iteration of HyCAPS when the migration happens).

        :param dump_file: Path where to dump the pickle file.
        """
        if self._migrations is not None:
            with open(dump_file, 'wb') as fo:
                pickle.dump(self._migrations, fo)
        else:
            logging.info("The tracking of the migration patterns was disabled during the run.\n"
                         "Set record_migration_patterns=True in the init function to track the fitnesses")
