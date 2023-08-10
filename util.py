import numpy as np
import random
import math
import math
import os
import more_itertools
import tqdm


def calculate_table_happiness(
    guests, affinities, importances, seated_apart_threshold, seated_together_threshold
):
    """Calculates the table happiness with its given seated guests. A table's happiness is the weighted sum of its
    guests' happiness where the weights are the guest importances.

    Args:
        guests (list): List of guest indices seated at the table.
        affinities (numpy.ndarray): Affinity matrix of all guests.
        importances (numpy.ndarray): Importances of all guests.
        seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies. If
            one of them have an affinity below this value, they are enemies and they are not to be seated at the same
            table.
        seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
            other, they must be seated at the same table.

    Returns:
        float: Table happiness.
    """
    total = 0
    for i in guests:
        if seated_together_threshold:
            for j, affinity in enumerate(affinities[i, :]):
                if (
                    affinity > seated_together_threshold
                    and affinities[j, i] > seated_together_threshold
                    and j not in guests
                ):  # Symmetrically friend but not in table
                    return -math.inf
        for j in guests:
            if i == j:
                continue
            if seated_apart_threshold:
                if affinities[i, j] < seated_apart_threshold:  # Enemy in table
                    return -math.inf
            total += importances[i] * affinities[i, j]
    return total


def calculate_tables_happiness(
    tables, affinities, importances, seated_apart_threshold, seated_together_threshold
):
    """Calculates the total happiness by calculating the happiness of each table and summing them.

    Args:
        tables (list): A list of tables (which are lists of guest indices).
        affinities (numpy.ndarray): Affinity matrix of all guests.
        importances (numpy.ndarray): Importances of all guests.
        seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies. If
            one of them have an affinity below this value, they are enemies and they are not to be seated at the same
            table.
        seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
            other, they must be seated at the same table.

    Returns:
        float: Total happiness.
    """
    total = 0
    for table in tables:
        happiness = calculate_table_happiness(
            table,
            affinities,
            importances,
            seated_apart_threshold,
            seated_together_threshold,
        )
        if happiness == -math.inf:
            return happiness
        total += happiness
    return total


def calculate_solution_happiness(
    solution, affinities, importances, seated_apart_threshold, seated_together_threshold
):
    """Calculates a solution's happiness. It is handy for when the solution is not a list of tables but a list of table
    indices.

    Args:
        solution (list): A list of table indices corresponding to guest indices. For example, the first value of the
            list indicates the table index of the first guest.
        affinities (numpy.ndarray): Affinity matrix of all guests.
        importances (numpy.ndarray): Importances of all guests.
        seated_apart_threshold (float): An affinity threshold used to decide if two guests are considered enemies. If
            one of them have an affinity below this value, they are enemies and they are not to be seated at the same
            table.
        seated_together_threshold (float): If two guests have an affinity value higher than this threshold for each
            other, they must be seated at the same table.

    Returns:
        float: Total happiness.
    """
    guest_tables = {guest_id: table for guest_id, table in enumerate(solution)}
    unique_tables = set(guest_tables.values())
    total = 0
    for table in unique_tables:
        guests = [k for k, v in guest_tables.items() if v == table]
        total += calculate_table_happiness(
            guests,
            affinities,
            importances,
            seated_apart_threshold,
            seated_together_threshold,
        )
    return total


def generate_network(
    n_guest,
    min_affinity,
    max_affinity,
    stranger_affinity,
    partner_affinity,
    min_importance,
    max_importance,
    affinity_symmetry_rate,
    affinity_asymmetry_deviation,
    stranger_rate,
    negative_affinity_rate,
    partner_rate,
):
    """Generates a guest network to solve using the given parameters.

    Args:
        n_guest (int): Number of guests
        min_affinity (float): Minimum affinity value between two guests.
        max_affinity (float): Maximum affinity value between two guests unless they are partners or strangers.
        stranger_affinity (float): The default affinity value for when a guest is a stranger to another guest.
        partner_affinity (float): The default affinity value for when two guests are partners.
        min_importance (float): Minimum guest importance.
        max_importance (float): Maximum guest importance
        affinity_symmetry_rate (float): The probability of two guests having the same affinity value for each other.
        affinity_asymmetry_deviation (float): The maximum deviation between two guests' affinities for each other when
            they have asymmetrical affinities.
        stranger_rate (float): The probability of a guest being a stranger to another guest.
        negative_affinity_rate (float): The probability of a guest having a negative affinity for another guest unless
            they are a stranger.
        partner_rate (float): The probability of a guest having a partner in the network.

    Returns:
        tuple: A tuple of guests, affinities, importances, and the parameters used to generate the network.
    """
    guests = list(range(n_guest))
    affinities = np.full((n_guest, n_guest), None)
    importances = [random.uniform(min_importance, max_importance) for _ in guests]
    for i in range(0, len(guests)):
        for j in range(i, len(guests)):
            partner = False
            if i == j:
                affinities[i, j] = 0
                continue
            elif i + 1 == j and (
                i == 0 or affinities[i, i - 1] != partner_affinity
            ):  # Trial for partnership
                partner_sample = random.uniform(0, 1)
                if partner_sample < partner_rate:  # Partner
                    partner = True
                    affinities[i, j] = partner_affinity
                    symmetry_sample = 0  # Forced symmetry

            if not partner:
                stranger_sample = random.uniform(0, 1)
                if stranger_sample < stranger_rate:  # Stranger
                    affinities[i, j] = stranger_affinity
                else:  # Not stranger
                    negative_affinity_sample = random.uniform(0, 1)
                    if negative_affinity_sample < negative_affinity_rate:  # Negative
                        affinities[i, j] = random.uniform(
                            min_affinity, -0.000001
                        )  # A very big negative value that is smaller than zero is used as the maximum negative value
                    else:  # Positive
                        affinities[i, j] = random.uniform(0.000001, max_affinity)

                symmetry_sample = random.uniform(0, 1)

            if (
                symmetry_sample < affinity_symmetry_rate and affinities[j, i] is None
            ):  # Symmetric
                affinities[j, i] = affinities[i, j]
            elif affinities[j, i] is None:  # Asymmetric within deviation bounds
                affinities[j, i] = random.uniform(
                    max(min_affinity, affinities[i, j] - affinity_asymmetry_deviation),
                    min(max_affinity, affinities[i, j] + affinity_asymmetry_deviation),
                )

    affinities = affinities.astype(float)
    return (
        guests,
        affinities,
        importances,
        {
            "n_guest": n_guest,
            "min_affinity": min_affinity,
            "max_affinity": max_affinity,
            "stranger_affinity": stranger_affinity,
            "partner_affinity": partner_affinity,
            "min_importance": min_importance,
            "max_importance": max_importance,
            "affinity_symmetry_rate": affinity_symmetry_rate,
            "affinity_asymmetry_deviation": affinity_asymmetry_deviation,
            "stranger_rate": stranger_rate,
            "negative_affinity_rate": negative_affinity_rate,
            "partner_rate": partner_rate,
        },
    )


def find_all_solutions(n_guest, n_seat, folder):
    """Finds all possible solutions for a given number of guests and seats using iteration rather than recursion. Its
    output is a .txt file under the given folder, naned using the parameters. Note that the output file can easily take
    gigabytes of space and require a long time to compute.

    Args:
        n_guest (int): Number of guests.
        n_seat (int): Number of seats per table.
        folder (str): A folder location. If the folder does not exist, it is created.

    Returns:
        str: Full path of the generated file.
    """

    guests = list(range(n_guest))
    tables = math.ceil(n_guest / n_seat)

    counter = 0
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{n_guest}_guests_{n_seat}_seats_solutions.txt")
    if os.path.isfile(filename):
        print("Solution already exists.")
        return filename
    with open(filename, "w") as file:
        for partition in tqdm.tqdm(more_itertools.set_partitions(guests, k=tables)):
            suitable = True
            for p in partition:
                if len(p) > n_seat:
                    # If the partition has more elements than the number of seats per table, it is ignored.
                    suitable = False
                    break
            if not suitable:
                continue

            file.write(str(partition) + "\n")
            counter += 1

    return os.path.abspath(filename)
