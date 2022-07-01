# Roster Master

A program that automatically generates a call roster for the month.

**Project Description:**

To generate a fair call roster that meets employee schedule requirements (e.g. requests and blocks) and team policy (e.g. minimum of 2 days between calls).
A fair roster is one that evenly distributes calls across normal and special days (i.e. weekends and holidays) for both the current month as well as taking into consideration historical roster records as best as possible.

There are two rosters to be generated: (i) Registra Roster (ii) Consultant Roster
There are three main employee groups: (i) Registras (ii) Associate-Consultants (iii) Consultants
When generating both rosters, registras and associate-consultants are under the Registra Roster while consultants and senior consultants are under the Consultant Roster.

As this is a side-project, the program uses the following open-source packages/software:
1. Pyomo: Python-based, open-source optimisation modelling language
2. GLPK: Open-source software package to solve large scale mixed integer programming

- Problem Objective
  - Minimise total variance between all employees' call credit
  - Satisfies all defined constraints 
<img width="176" alt="Screenshot 2022-04-13 at 9 23 05 PM" src="https://user-images.githubusercontent.com/54014264/176854536-d576aa6f-4561-4e24-a111-7f9fc123bfcd.png">
- Basic Constraints
  - Employee "Blocked" constraint: Employees cannot be assigned on a day they have blocked
  - Employee "Requested" constraint: Employees must be assigned on a day they have requested
  - Daily Quota constraint: There must be at least one employee assigned call every day
<img width="558" alt="Screenshot 2022-07-01 at 4 17 36 PM" src="https://user-images.githubusercontent.com/54014264/176854706-20ebb176-4815-4fbc-9252-ec67f18f81bf.png">
- Special Constraints (Specific to sample user's user case - may not be applicable for general roster generation)
  - 2-Day constraint: Can only be assigned calls that are at least 2 days apart (unless they opt out)
  - Special Day constraint: All employees must be assigned at least 1 special day in the month, can be adjusted to only apply for those not in the exclusion list.
  - Exclusion List and Max Calls: Can exclude employees from the optimisation of total variance and set the maximum number of calls they can be assigned
  - Call Credits equalised over time: Difference between employees' total call credits for the month and the monthly mode are archived and rolled over to subsequent months
  - AC Requests: All employees in consultant roster to be on call with AC at least once a month
<img width="642" alt="Screenshot 2022-07-01 at 4 18 22 PM" src="https://user-images.githubusercontent.com/54014264/176854790-b2b9d2cf-c973-468f-802c-78b310d88320.png">


**Limitations:**

1. User is required to have some background in running the python script (having issues compiling an executable that can run pyomo and glpk)
2. User is required to vet the data input (i.e. employee schedule requirements) to ensure it does not violate key model requirements
3. As this project's main objective is to minimize the time and effort spent on roster scheduling, the program prioritizes feasibility over optimality and will choose the best feasible solution if it is taking too long to find an optimal solution.
