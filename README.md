# Metaheuristic Seating Plan Optimization

## Problem statement

Assume that you need to organize a seating plan for a wedding where there are $x$ guests and $y$ tables with $z$ seats each. Guests' satisfaction (also referred to as happiness) depends on the people with whom they are seated, and you need to maximize the total satisfaction by seating people based on their affinities with other guests. It can be a daunting task especially with larger populations, and it is considered NP-hard.

Luckily, we can tackle this problem using combinatorial optimization. This repository includes _some_ of the code I wrote for this problem as my Combinatorial Algorithms course project at the University of Ottawa. I implemented Simulated Annealing, the Great Deluge, and Record-to-Record Travel algorithms, which are metaheuristic algorithms. I also implemented exhaustive, greedy, and random solvers for comparison purposes.

Existing solutions predominantly have the following deficiencies:
*   Affinities are treated symmetrically, but this is not the case in real life. Someone you like may not like you as much, or at least not as much as they like someone else. They may not even know that you exist. 
*   Affinities are treated as discrete categories, but it may not be as simple as "friends" and "enemies."
*   Strangers have no affinity with each other, but people may not like to be seated  with strangers.
*   All guests are treated equally, but people care more about their closest friends and relatives.

## My approach

I expanded on the base problem as follows:

*   Affinities are not necessarily symmetrical. They are represented as floats within a given range (between -1.0 and +1.0), and strangers have a small negative affinity (-0.1).
*   Guests (nodes) have an importance factor which affects the total happiness when they are aggregated. So, some guests' satisfaction is more important to us.
*   There are some hard constraints about when two people must and must not be seated together. For example, you may want to keep people who have very negative affinities (< -0.75) for each other separate, or ensure people who have very high affinities (> 0.75) for each other are seated together. However, these hard constraints can be hard to satisfy depending on their strictness and the dataset complexity (and it is not that hard to generate some impossible scenarios). For this reason, I mostly focused on the soft constraint (total satisfaction) for my project. 

I focused on metaheuristic algorithms that can find a solution that is either optimal or very close to it while being orders of magnitudes faster than an exhaustive solution. Simulated Annealing is a very famous algorithm. The Great Deluge and Record-to-Record Travel (threshold-accepting algorithms) were invented and published by Gunter Dueck. As claimed by the author, I found that these two algorithms are much better than Simulated Annealing in terms of performance, parameter optimization, and intuitiveness. 

All solvers return the solution in a table index format where the first value of the solution indicates the table index of the first guest and so on. Constraints and dataset characteristics are customizable through the solvers' and the dataset generator's parameters. The modules have docstrings and an example Jupyter notebook with a very simple example is provided. The exhaustive solution file or folder is not provided, but both can be easily generated using the given function, as shown in the example.

## Limitations

*   All tables must have the same amount of seats. Tables can have empty seats, but the maximum number of allowed tables is the ceiling division of guests by seats per table (otherwise the problem is trivial).
*   Metaheuristic algorithms are not guaranteed to find the optimal (or even valid, depending on the constraints) solution. However, the algorithms implemented as alternatives to Simulated Annealing were quite successful in general. 
*   This approach is not the best for hard constraint-focused approaches. Also, some additional approaches such as representing couples as a single node that takes up two seats (averaging their affinities with other guests?) may be helpful.
*   Generating all possible solutions for a complex case can easily take a long time and many gigabytes of space. For such cases, you may want to stick to the metaheuristic solutions without having a ground truth.