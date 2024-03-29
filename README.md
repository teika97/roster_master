# Roster Master

A program that automatically generates a call roster for the month.

## Project Description

To generate a fair call roster that meets employee schedule requirements (e.g. requests and blocks) and team policy (e.g. minimum of 2 days between calls).
A fair roster is one that evenly distributes calls across normal and special days (i.e. weekends and holidays) for both the current month as well as taking into consideration historical roster records as best as possible.

There are two rosters to be generated: (i) Registra Roster (ii) Consultant Roster and three main employee groups: (a) Registras (b) Associate-Consultants (b) Consultants. When generating both rosters, registras and associate-consultants are under the Registra Roster while consultants and senior consultants are under the Consultant Roster.

As this is a side-project, the program uses the following open-source packages/software:
1. Pyomo: Python-based, open-source optimisation modelling language
2. GLPK: Open-source software package to solve large scale mixed integer programming

### Problem Objective
<img width="698" alt="Screenshot 2022-07-01 at 4 16 39 PM" src="https://user-images.githubusercontent.com/54014264/176854885-9c538839-9956-48e4-82bf-9aaf51f11c27.png">  

- Minimise total variance between all employees' call credit
- Satisfies all defined constraints 
  
### Basic Constraints
<img width="558" alt="Screenshot 2022-07-01 at 4 17 36 PM" src="https://user-images.githubusercontent.com/54014264/176854706-20ebb176-4815-4fbc-9252-ec67f18f81bf.png">

- **Employee "Blocked" constraint:** Employees cannot be assigned on a day they have blocked
- **Employee "Requested" constraint:** Employees must be assigned on a day they have requested
- **Daily Quota constraint:** There must be at least one employee assigned call every day
  
### Special Constraints (Specific to sample user's user case - may not be applicable for all)
<img width="642" alt="Screenshot 2022-07-01 at 4 18 22 PM" src="https://user-images.githubusercontent.com/54014264/176854790-b2b9d2cf-c973-468f-802c-78b310d88320.png">

- **2-Day constraint:** Can only be assigned calls that are at least 2 days apart (unless they opt out)
- **Special Day constraint:** All employees must be assigned at least 1 special day in the month (can be adjusted to exclude those in exclusion list)
- **Exclusion List and Max Calls:** Exclude select employees from the optimisation of total variance and set the maximum number of calls they can be assigned
- **Call Credits equalised over time:** Difference between employees' total call credits for the month and the monthly mode are archived and rolled over to subsequent months
- **AC Requests:** All employees in consultant roster to be on call with AC at least once a month

### Final Output
Script outputs a single excel file:
1. "IterationData_Reg" Tab:
    - Employee schedule requirements
    - 30 iterations of the registra roster (each to be paired with the corresponding iteration in the consultant tab)
2. "IterationData_Consultant" Tab:
    - Employee schedule requirements
    - 30 iterations of the consultant roster (each to be paired with corresponding iteration in the registra tab)
3. "Roster" Tab:
    - Registra and Consultant rosters for all 30 iterations in format preferred by Admin

## Limitations

1. User is required to have some background in running the python script (having issues compiling an executable that can run pyomo and glpk)
2. User is required to vet the data input (i.e. employee schedule requirements) to ensure it does not violate key model requirements
3. As this project's main objective is to minimize the time and effort spent on roster scheduling, the program prioritizes feasibility over optimality and will choose the best feasible solution if it is takes too long to find an optimal solution.
