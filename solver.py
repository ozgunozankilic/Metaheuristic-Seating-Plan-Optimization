import math
import util
import numpy as np
import random
import tqdm
import time
import ast


class GreatDelugeSolver:
    """A Great Deluge (Gunter Dueck) implementation for seating optimization."""

    def __init__(
        self,
        guests,
        affinities,
        importances,
        seated_apart_threshold,
        seated_together_threshold,
        n_seat,
        water_level=0,
        rain_speed="auto",
        max_iterations=100000,
        max_iterations_without_improvement=None,
        silent=False,
    ):
        """Initializes the Great Deluge solver using the given parameters and problem.

        Args:
            guests (list): A list of guest IDs (from 0 to n-1).
            affinities (numpy.ndarray): A numpy array that essentially keeps the interpersonal affinities in a matrix
                format.
            importances (numpy.ndarray): A numpy array that keeps the guest importance values.
            seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies.
                If one of them have an affinity below this value, they are enemies and they are not to be seated at the
                same table.
            seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
                other, they must be seated at the same table.
            n_seat (int): The number of seats per table.
            water_level (float, optional): The water level of the algorithm. It indicates the initial quality threshold.
                Defaults to 0.
            rain_speed (float|"auto", optional): Rain speed value of the algorithm. Essentially, it indicates the amount
                of increase for the threshold when a new solution is accepted. It automatically finds the best value
                when set to "auto" based on Dueck's suggestion in his paper. Defaults to "auto".
            max_iterations (int, optional): The maximum number of iterations the solver can try optimizing. Defaults to
                100000.
            max_iterations_without_improvement (int|None, optional): The maximum number of iterations the solver can go
                without seeing an improvement. If None, it does not stop until it hits max_iterations. Defaults to None.
            silent (bool, optional): Indicates whether the solver must run silently. Defaults to False.

        Returns:
            None
        """
        self.guests = set(guests)
        self.n_guest = len(guests)
        self.affinities = affinities
        self.importances = importances
        self.seated_apart_threshold = seated_apart_threshold
        self.seated_together_threshold = seated_together_threshold
        self.n_seat = n_seat
        self.n_table = math.ceil(self.n_guest / self.n_seat)
        self.remaining_guests = self.guests.copy()
        self.guest_tables = {guest: None for guest in self.guests}
        self.best_happiness = -math.inf
        self.best_arrangement = None
        self.water_level = water_level
        self.rain_speed = rain_speed
        self.iterations = 0
        self.max_iterations = max_iterations
        if max_iterations_without_improvement is None:
            self.max_iterations_without_improvement = self.max_iterations
        else:
            self.max_iterations_without_improvement = max_iterations_without_improvement
        self.execution_time = None
        self.happiness_history = []
        self.threshold_history = []
        self.silent = silent

    def solve(self, initial_solution=None):
        """Solves the previously given problem.

        Args:
            initial_solution (list|None, optional): A list of table indices corresponding to guests. For example, the
                first index of this list corresponds to the first guest. If None, a random initial solution is created.
                Defaults to None.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        if initial_solution is None:
            # Create a random initial solution
            initial_solution = []
            for table in range(self.n_table):
                initial_solution.extend([table] * self.n_seat)

            random.shuffle(initial_solution)

        start = time.time()
        self.best_happiness = util.calculate_solution_happiness(
            initial_solution[: self.n_guest],
            self.affinities,
            self.importances,
            self.seated_apart_threshold,
            self.seated_together_threshold,
        )
        self.best_arrangement = initial_solution
        solution = initial_solution.copy()
        last_happiness = self.best_happiness
        self.iterations = 0
        iterations_without_improvement = 0
        progress_bar = tqdm.tqdm(total=self.max_iterations, disable=self.silent)
        while (
            self.iterations < self.max_iterations
            and iterations_without_improvement < self.max_iterations_without_improvement
        ):
            self.iterations += 1
            progress_bar.update(1)
            id_1, id_2 = random.sample(self.guests, 2)
            while (id_1 >= self.n_guest and id_2 >= self.n_guest) or (
                solution[id_1] == solution[id_2]
            ):
                # If both seats are empty or both seats are from the same table, resample
                id_1, id_2 = random.sample(self.guests, 2)

            new_solution = solution.copy()
            new_solution[id_1], new_solution[id_2] = (
                new_solution[id_2],
                new_solution[id_1],
            )
            new_happiness = util.calculate_solution_happiness(
                new_solution[: self.n_guest],
                self.affinities,
                self.importances,
                self.seated_apart_threshold,
                self.seated_together_threshold,
            )
            if new_happiness > self.best_happiness:
                self.best_happiness = new_happiness
                self.best_arrangement = new_solution

            if new_happiness > self.water_level:
                iterations_without_improvement = 0
                solution = new_solution
                last_happiness = new_happiness
                if self.rain_speed == "auto":
                    self.water_level += (new_happiness - self.water_level) * 0.009
                else:
                    self.water_level += self.rain_speed
            else:
                iterations_without_improvement += 1

            self.happiness_history.append(last_happiness)
            self.threshold_history.append(self.water_level)

        end = time.time()
        self.execution_time = end - start

        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement[: self.n_guest],
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }

    def return_last_solution(self):
        """Returns the last obtained solution's details.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement[: self.n_guest],
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }


class RecordToRecordSolver:
    """A Record-To-Record Travel (Gunter Dueck) implementation for seating optimization."""

    def __init__(
        self,
        guests,
        affinities,
        importances,
        seated_apart_threshold,
        seated_together_threshold,
        n_seat,
        deviation,
        max_iterations=100000,
        max_iterations_without_improvement=None,
        silent=False,
    ):
        """Initializes the Great Deluge solver using the given parameters and problem.

        Args:
            guests (list): A list of guest IDs (from 0 to n-1).
            affinities (numpy.ndarray): A numpy array that essentially keeps the interpersonal affinities in a matrix
                format.
            importances (numpy.ndarray): A numpy array that keeps the guest importance values.
            seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies.
                If one of them have an affinity below this value, they are enemies and they are not to be seated at the
                same table.
            seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
                other, they must be seated at the same table.
            n_seat (int): The number of seats per table.
            deviation (float): Allowed deviation amount of the algorithm. A new solution's quality must be at least
                record quality - deviation (within the allowed deviation range) to be accepted.
            max_iterations (int, optional): The maximum number of iterations the solver can try optimizing. Defaults to
                100000.
            max_iterations_without_improvement (int|None, optional): The maximum number of iterations the solver can go
                without seeing an improvement. If None, it does not stop until it hits max_iterations. Defaults to None.
            silent (bool, optional): Indicates whether the solver must run silently. Defaults to False.

        Returns:
            None
        """
        self.guests = set(guests)
        self.n_guest = len(guests)
        self.affinities = affinities
        self.importances = importances
        self.seated_apart_threshold = seated_apart_threshold
        self.seated_together_threshold = seated_together_threshold
        self.n_seat = n_seat
        self.n_table = math.ceil(self.n_guest / self.n_seat)
        self.remaining_guests = self.guests.copy()
        self.guest_tables = {guest: None for guest in self.guests}
        self.best_happiness = -math.inf
        self.best_arrangement = None
        self.deviation = deviation
        self.iterations = 0
        self.max_iterations = max_iterations
        if max_iterations_without_improvement is None:
            self.max_iterations_without_improvement = self.max_iterations
        else:
            self.max_iterations_without_improvement = max_iterations_without_improvement
        self.happiness_history = []
        self.threshold_history = []
        self.silent = silent

    def solve(self, initial_solution=None):
        """Solves the previously given problem.

        Args:
            initial_solution (list|None, optional): A list of table indices corresponding to guests. For example, the
                first index of this list corresponds to the first guest. If None, a random initial solution is created.
                Defaults to None.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        if initial_solution is None:
            # Create a random initial solution
            initial_solution = []
            for table in range(self.n_table):
                initial_solution.extend([table] * self.n_seat)

            random.shuffle(initial_solution)

        start = time.time()
        self.best_happiness = util.calculate_solution_happiness(
            initial_solution[: self.n_guest],
            self.affinities,
            self.importances,
            self.seated_apart_threshold,
            self.seated_together_threshold,
        )
        self.best_arrangement = initial_solution
        solution = initial_solution.copy()
        last_happiness = self.best_happiness
        self.iterations = 0
        iterations_without_improvement = 0
        progress_bar = tqdm.tqdm(total=self.max_iterations, disable=self.silent)
        threshold = self.best_happiness - self.deviation
        while (
            self.iterations < self.max_iterations
            and iterations_without_improvement < self.max_iterations_without_improvement
        ):
            self.iterations += 1
            progress_bar.update(1)
            id_1, id_2 = random.sample(self.guests, 2)
            while (id_1 >= self.n_guest and id_2 >= self.n_guest) or (
                solution[id_1] == solution[id_2]
            ):
                # If both seats are empty or both seats are from the same table, resample
                id_1, id_2 = random.sample(self.guests, 2)

            new_solution = solution.copy()
            new_solution[id_1], new_solution[id_2] = (
                new_solution[id_2],
                new_solution[id_1],
            )
            new_happiness = util.calculate_solution_happiness(
                new_solution[: self.n_guest],
                self.affinities,
                self.importances,
                self.seated_apart_threshold,
                self.seated_together_threshold,
            )
            if new_happiness > self.best_happiness:
                self.best_happiness = new_happiness
                self.best_arrangement = new_solution
                threshold = self.best_happiness - self.deviation

            if new_happiness > threshold:
                iterations_without_improvement = 0
                solution = new_solution
                last_happiness = new_happiness
            else:
                iterations_without_improvement += 1

            self.happiness_history.append(last_happiness)
            self.threshold_history.append(threshold)

        end = time.time()
        self.execution_time = end - start

        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement[: self.n_guest],
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }

    def return_last_solution(self):
        """Returns the last obtained solution's details.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement[: self.n_guest],
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }


class SimulatedAnnealingSolver:
    """A Simulated Annealing implementation. It allows different types of cooling schedules."""

    def __init__(
        self,
        guests,
        affinities,
        importances,
        seated_apart_threshold,
        seated_together_threshold,
        n_seat,
        temperature,
        alpha,
        max_iterations=100000,
        max_iterations_without_improvement=None,
        silent=False,
    ):
        """Initializes the Great Deluge solver using the given parameters and problem.

        Args:
             guests (list): A list of guest IDs (from 0 to n-1).
            affinities (numpy.ndarray): A numpy array that essentially keeps the interpersonal affinities in a matrix
                format.
            importances (numpy.ndarray): A numpy array that keeps the guest importance values.
            seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies.
                If one of them have an affinity below this value, they are enemies and they are not to be seated at the
                same table.
            seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
                other, they must be seated at the same table.
            n_seat (int): The number of seats per table.
            temperature (float): The initial temperature of the algorithm.
            alpha (float|int|"auto"): The cooling factor of the algorithm. It is automatically set if set to "auto"
                through dividing the temperature by the current number of iterations. If it is greater than or equal to
                1, the alpha is used linearly by subtracting it from the temperature. Otherwise, it is used as usual (it
                is multiplied with the temperature).
            max_iterations (int, optional): The maximum number of iterations the solver can try optimizing. Defaults to
                100000.
            max_iterations_without_improvement (int|None, optional): The maximum number of iterations the solver can go
                without seeing an improvement. If None, it does not stop until it hits max_iterations. Defaults to None.
            silent (bool, optional): Indicates whether the solver must run silently. Defaults to False.

        Returns:
            None
        """
        self.guests = set(guests)
        self.n_guest = len(guests)
        self.affinities = affinities
        self.importances = importances
        self.seated_apart_threshold = seated_apart_threshold
        self.seated_together_threshold = seated_together_threshold
        self.n_seat = n_seat
        self.n_table = math.ceil(self.n_guest / self.n_seat)
        self.remaining_guests = self.guests.copy()
        self.guest_tables = {guest: None for guest in self.guests}
        self.best_happiness = -math.inf
        self.best_arrangement = None
        self.temperature = temperature
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.iterations = 0
        if max_iterations_without_improvement is None:
            self.max_iterations_without_improvement = self.max_iterations
        else:
            self.max_iterations_without_improvement = max_iterations_without_improvement
        self.happiness_history = []
        self.temperature_history = []
        self.execution_time = None
        self.silent = silent

    def solve(self, initial_solution=None):
        """Solves the previously given problem.

        Args:
            initial_solution (list|None, optional): A list of table indices corresponding to guests. For example, the
                first index of this list corresponds to the first guest. If None, a random initial solution is created.
                Defaults to None.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        if initial_solution is None:
            # Create a random initial solution
            initial_solution = []
            for table in range(self.n_table):
                initial_solution.extend([table] * self.n_seat)

            random.shuffle(initial_solution)

        start = time.time()
        self.best_happiness = util.calculate_solution_happiness(
            initial_solution[: self.n_guest],
            self.affinities,
            self.importances,
            self.seated_apart_threshold,
            self.seated_together_threshold,
        )
        self.best_arrangement = initial_solution
        solution = initial_solution.copy()
        last_happiness = self.best_happiness
        self.iterations = 0
        iterations_without_improvement = 0
        progress_bar = tqdm.tqdm(total=self.max_iterations, disable=self.silent)
        while (
            self.iterations < self.max_iterations
            and self.temperature > 0
            and iterations_without_improvement < self.max_iterations_without_improvement
        ):
            self.iterations += 1
            progress_bar.update(1)
            id_1, id_2 = random.sample(self.guests, 2)
            while id_1 >= self.n_guest and id_2 >= self.n_guest:
                id_1, id_2 = random.sample(self.guests, 2)

            new_solution = solution.copy()
            new_solution[id_1], new_solution[id_2] = (
                new_solution[id_2],
                new_solution[id_1],
            )
            new_happiness = util.calculate_solution_happiness(
                new_solution[: self.n_guest],
                self.affinities,
                self.importances,
                self.seated_apart_threshold,
                self.seated_together_threshold,
            )
            if new_happiness > self.best_happiness:
                self.best_happiness = new_happiness
                self.best_arrangement = new_solution

            happiness_diff = new_happiness - last_happiness
            if happiness_diff > 0:
                iterations_without_improvement = 0
                solution = new_solution
                last_happiness = new_happiness
            else:
                iterations_without_improvement += 1
                sample = random.uniform(0.0, 1.0)
                if sample < math.exp(happiness_diff / self.temperature):
                    solution = new_solution
                    last_happiness = new_happiness

            self.happiness_history.append(last_happiness)
            if self.alpha == "auto":
                self.temperature /= self.iterations
            elif self.alpha >= 1:
                self.temperature -= self.alpha
            else:
                self.temperature *= self.alpha

            self.temperature_history.append(self.temperature)
        end = time.time()
        self.execution_time = end - start

        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement[: self.n_guest],
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }

    def return_last_solution(self):
        """Returns the last obtained solution's details.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement[: self.n_guest],
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }


class ExhaustiveSolver:
    """An exhaustive solver that tries all possible solutions and finds the best one. Since recursion does not work well
    especially with more complex networks, it uses the file that has all possible solutions and finds the one that has
    the highest happiness.
    """

    def __init__(
        self,
        guests,
        affinities,
        importances,
        seated_apart_threshold,
        seated_together_threshold,
        n_seat,
        possible_solutions_file,
    ):
        """Initializes the solver with the given parameters.

        Args:
            guests (list): A list of guest IDs (from 0 to n-1).
            affinities (numpy.ndarray): A numpy array that essentially keeps the interpersonal affinities in a matrix
                format.
            importances (numpy.ndarray): A numpy array that keeps the guest importance values.
            seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies.
                If one of them have an affinity below this value, they are enemies and they are not to be seated at the
                same table.
            seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
                other, they must be seated at the same table.
            n_seat (int): The number of seats per table.
            possible_solutions_file (str): Location of the file that stores all possible solutions.
        """
        self.guests = set(guests)
        self.n_guest = len(guests)
        self.affinities = affinities
        self.importances = importances
        self.seated_apart_threshold = seated_apart_threshold
        self.seated_together_threshold = seated_together_threshold
        self.n_seat = n_seat
        self.n_table = math.ceil(self.n_guest / self.n_seat)
        self.best_happiness = -math.inf
        self.best_arrangement = None
        self.possible_solutions_file = possible_solutions_file
        self.iterations = 0
        self.execution_time = None

    def solve(self):
        """Solves the previously given problem.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        start = time.time()
        self.iterations = 0
        with open(self.possible_solutions_file, "r") as file:
            for line in tqdm.tqdm(file):
                self.iterations += 1
                tables = ast.literal_eval(line)  # A safer alternative to eval
                happiness = util.calculate_tables_happiness(
                    tables,
                    self.affinities,
                    self.importances,
                    self.seated_apart_threshold,
                    self.seated_together_threshold,
                )

                if happiness > self.best_happiness:
                    self.best_arrangement = tables
                    self.best_happiness = happiness

        if self.best_arrangement is not None:
            solution = [None] * self.n_guest
            for table_id, table in enumerate(self.best_arrangement):
                for guest in table:
                    solution[guest] = table_id

            self.best_arrangement = solution

        end = time.time()
        self.execution_time = end - start

        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement,
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }

    def return_last_solution(self):
        """Returns the last obtained solution's details.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement,
            "Iterations": self.iterations,
            "Execution time": self.execution_time,
        }


class GreedySolver:
    """A greedy solver for comparison. It starts with the most important person, and finds the person that would
    maximize their happiness. After seating this person, it then finds the person that would maximize the first two
    people's happiness. This process is iterated until the table is full, and it starts this process again with the most
    important non-seated person until everyone is seated. Obviously, it is far from being optimal.
    """

    def __init__(
        self,
        guests,
        affinities,
        importances,
        seated_apart_threshold,
        seated_together_threshold,
        n_seat,
    ):
        """Initializes the solver with the given parameters.

        Args:
            guests (list): A list of guest IDs (from 0 to n-1).
            affinities (numpy.ndarray): A numpy array that essentially keeps the interpersonal affinities in a matrix
                format.
            importances (numpy.ndarray): A numpy array that keeps the guest importance values.
            seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies.
                If one of them have an affinity below this value, they are enemies and they are not to be seated at the
                same table.
            seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
                other, they must be seated at the same table.
            n_seat (int): The number of seats per table.
        """
        self.guests = set(guests)
        self.n_guest = len(guests)
        self.affinities = affinities
        self.importances = importances
        self.seated_apart_threshold = seated_apart_threshold
        self.seated_together_threshold = seated_together_threshold
        self.n_seat = n_seat
        self.n_table = math.ceil(self.n_guest / self.n_seat)
        self.remaining_guests = self.guests.copy()
        self.tables = []
        self.guest_tables = {guest: None for guest in self.guests}
        self.best_happiness = -math.inf
        self.best_arrangement = None
        self.execution_time = None

    def solve(self):
        """Solves the previously given problem.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        start = time.time()
        if len(self.remaining_guests) < 1:
            return self.best_arrangement

        current_table = len(self.tables)
        progress_bar = tqdm.tqdm(total=self.n_table)
        while len(self.remaining_guests) > 0:
            if len(self.tables) <= current_table:
                self.tables.append([])
            if len(self.tables[current_table]) == 0:
                # New table, start with the most important person
                max_importance = 0
                most_important_guest = None
                for guest in self.guests:
                    if (
                        guest in self.remaining_guests
                        and self.importances[guest] > max_importance
                    ):
                        max_importance = self.importances[guest]
                        most_important_guest = guest

                self.remaining_guests.remove(most_important_guest)
                self.tables[current_table].append(most_important_guest)

            elif len(self.tables[current_table]) >= self.n_seat:
                # The table is full, create a new table
                current_table += 1
                progress_bar.update(1)
                self.tables.append([])

            else:
                # Add guests to the existing table
                while (
                    len(self.tables[current_table]) < self.n_seat
                    and len(self.remaining_guests) > 0
                ):
                    max_affinity = -math.inf
                    max_affinity_guest = None
                    for guest in self.remaining_guests:
                        if (
                            guest != most_important_guest
                            and util.calculate_table_happiness(
                                self.tables[current_table] + [guest],
                                self.affinities,
                                self.importances,
                                self.seated_apart_threshold,
                                self.seated_together_threshold,
                            )
                            >= max_affinity
                        ):
                            max_affinity = self.affinities[most_important_guest, guest]
                            max_affinity_guest = guest

                    if (
                        max_affinity_guest is None
                    ):  # Just in case, it should not be None
                        max_affinity_guest = self.remaining_guests.pop()
                    else:
                        self.remaining_guests.remove(max_affinity_guest)

                    self.tables[current_table].append(max_affinity_guest)

        for table_id, table_guests in enumerate(self.tables):
            for guest in table_guests:
                self.guest_tables[guest] = table_id

        self.best_arrangement = [self.guest_tables[i] for i in range(self.n_guest)]
        self.best_happiness = util.calculate_solution_happiness(
            self.best_arrangement,
            self.affinities,
            self.importances,
            self.seated_apart_threshold,
            self.seated_together_threshold,
        )

        end = time.time()
        self.execution_time = end - start

        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement,
            "Iterations": None,
            "Execution time": self.execution_time,
        }

    def return_last_solution(self):
        """Returns the last obtained solution's details.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement,
            "Iterations": None,
            "Execution time": self.execution_time,
        }


class RandomSolver:
    """Generates a totally random solution for comparison."""

    def __init__(
        self,
        guests,
        affinities,
        importances,
        seated_apart_threshold,
        seated_together_threshold,
        n_seat,
    ):
        """Initializes the solver with the given parameters.

        Args:
            guests (list): A list of guest IDs (from 0 to n-1).
            affinities (numpy.ndarray): A numpy array that essentially keeps the interpersonal affinities in a matrix
                format.
            importances (numpy.ndarray): A numpy array that keeps the guest importance values.
            seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies.
                If one of them have an affinity below this value, they are enemies and they are not to be seated at the
                same table.
            seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
                other, they must be seated at the same table.
            n_seat (int): The number of seats per table.
        """
        self.guests = set(guests)
        self.n_guest = len(guests)
        self.affinities = affinities
        self.importances = importances
        self.seated_apart_threshold = seated_apart_threshold
        self.seated_together_threshold = seated_together_threshold
        self.n_seat = n_seat
        self.n_table = math.ceil(self.n_guest / self.n_seat)
        self.remaining_guests = self.guests.copy()
        self.tables = []
        self.guest_tables = {guest: None for guest in self.guests}
        self.best_happiness = -math.inf
        self.best_arrangement = None
        self.execution_time = 0

    def solve(self):
        """Solves the previously given problem.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        start = time.time()
        initial_solution = []
        for table in range(self.n_table):
            initial_solution.extend([table] * self.n_seat)

        random.shuffle(initial_solution)
        self.best_arrangement = initial_solution[: self.n_guest]
        self.best_happiness = util.calculate_solution_happiness(
            self.best_arrangement,
            self.affinities,
            self.importances,
            self.seated_apart_threshold,
            self.seated_together_threshold,
        )
        end = time.time()
        self.execution_time = end - start
        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement,
            "Iterations": None,
            "Execution time": self.execution_time,
        }

    def return_last_solution(self):
        """Returns the last obtained solution's details.

        Returns:
            dict: A dictionary that has the solution's happiness, seating arrangement, iterations, and execution time.
        """
        return {
            "Best happiness": self.best_happiness,
            "Best arrangement": self.best_arrangement,
            "Iterations": None,
            "Execution time": self.execution_time,
        }
