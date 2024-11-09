import matplotlib.pyplot as plt
import numpy as np
from main import *


def run_tests(solvers: list[Solver], map_files: list[str], epochs: int):
    # Store results in a nested dictionary: results[solver_name][map_name][metric] = value
    results = {}
    # generate list of problems
    maps_problems = []
    io_handler = IOHandler()

    for map_file in map_files:
        io_handler.set_input_file_name(map_file)
        maze, rock_weights = io_handler.parse()

        problem = Problem(Environment(maze, rock_weights))
        maps_problems.append((map_file, problem))

    for solver in solvers:
        solver_name = solver.algorithm_name
        results[solver_name] = {}
        for map_name, problem in maps_problems:
            results[solver_name][map_name] = {
                "runtime": [],
                "memory": [],
                "nodes_generated": [],
                "steps": [],
                "cost": [],
            }

    for _ in range(epochs):
        print(f"EPOCH {_ + 1}/{epochs}")
        print("=========================================================")
        for map_name, problem in maps_problems:
            print(f"MAP: {map_name}")
            print("-----------------------------------------------------------")
            for solver in solvers:
                solver_name = solver.algorithm_name

                # Solve the problem
                solver.solve_and_measure(problem)

                runtime = solver.get_time_taken()
                memory = solver.get_memory_used()
                nodes_generated = solver.nodes_generated
                steps = solver.steps
                cost = solver.total_cost

                print(solver.output_metrics())

                # Store the metrics
                results[solver_name][map_name]["runtime"].append(runtime)
                results[solver_name][map_name]["memory"].append(memory)
                results[solver_name][map_name]["nodes_generated"].append(
                    nodes_generated
                )
                results[solver_name][map_name]["steps"].append(steps)
                results[solver_name][map_name]["cost"].append(cost)
            print("-----------------------------------------------------------")
        print("=========================================================")

    # Compute average metrics
    avg_results = {}
    for solver_name in results:
        avg_results[solver_name] = {}
        for map_name in results[solver_name]:
            avg_metrics = {}
            for metric in results[solver_name][map_name]:
                avg_value = np.mean(results[solver_name][map_name][metric])
                avg_metrics[metric] = avg_value
            avg_results[solver_name][map_name] = avg_metrics

    # Plotting
    metrics = ["runtime", "memory", "nodes_generated", "steps", "cost"]
    for metric in metrics:
        plt.figure()
        for solver_name in avg_results:
            map_names = list(avg_results[solver_name].keys())
            values = [
                avg_results[solver_name][map_name][metric] for map_name in map_names
            ]
            plt.plot(map_names, values, marker="o", label=solver_name)
        plt.xlabel("Maps")
        plt.ylabel(metric.capitalize())
        plt.title(f"Average {metric.capitalize()} per Map")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    # Example usage:
    solvers = [DFSolver(), BFSolver(), UCSolver(), AStarSolver()]
    map_files = [
        "input-01.txt",
        "input-02.txt",
        "input-03.txt",
        "input-04.txt",
        "input-05.txt",
        "input-06.txt",
        "input-07.txt",
        "input-08.txt",
        "input-10.txt",
    ]  # Replace with actual map file paths
    epochs = 1
    run_tests(solvers, map_files, epochs)


if __name__ == "__main__":
    main()
