# roster_master

To design a program to automate the generation of the call roster

Project Title:
Roster Master

Project Description:
To generate a fair call roster that meets employee schedule requirements (e.g. requests and blocks) and team policy (e.g. minimum of 2 days between calls).
A fair roster is one that evenly distributes calls across normal and special days (i.e. weekends and holidays) for both the current month as well as taking into consideration historical roster records as best as possible.

As this is a side-project, the program uses the following open-source packages/software:
1. Pyomo: Python-based, open-source optimisation modelling language
2. GLPK: Open-source software package to solve large scale mixed integer programming

Limitations:
1. User is required to have some background in running the python script (having issues compiling an executable that can run pyomo and glpk)
2. User is required to vet the data input (i.e. employee requirements) to ensure it does not violate key model requirements
3. As this project's main objective is to minimize the time and effort spent on roster scheduling, the program prioritizes feasibility over optimality and will choose the best feasible solution if it is taking too long to find an optimal solution.
