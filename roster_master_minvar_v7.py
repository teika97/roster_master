# -*- coding: utf-8 -*-
# Using min. total call credit variance
import os
import re
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import calendar
import holidays
import datetime as dt
from datetime import datetime, timedelta

#%% Set programme run settings

test_flag = True
model_time_cutoff = 10
model_iterations = 10

req_folder_name = "Sample_Files"
month_regex = "((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) 20[2-9][0-9])"
req_file_regex_c = "Call requests \(Consultant\) - " + month_regex + ".csv"
req_file_regex_r = "Call requests \(Reg\) - " + month_regex + ".csv"
req_date_regex = "%b %Y"

# Legend for Block Types: AL,TL,X,NSL,OIL,ML,PL,FCL,CCL,KKH,PDL
# Legend for Request: R, CR
blocked_types = ['AL','TL','X','NSL','OIL','ML','PL','FCL','CCL','KKH','PDL']
requested_types = ['R','CR']


#%% Read data for model input

"""Gets file date from the file

Args : 
    file_name (str): Name of the file
    file_regex (str): Regex expression to specify file name pattern to extract file_date
    date_regex (str): Regex expression to specify date pattern in file_name
    return_type (str): Return type of file_date as string or date-time object
    
Returns :
    date as date-time object or string as extracted from file_name
"""
def get_date(file_name, file_regex, date_regex, return_type = "dt"):
    
    file_date = re.search(file_regex, file_name).group(1)
    
    if (return_type == "str"):
        return file_date
    else:
        return dt.datetime.strptime(file_date, date_regex)
    
    
"""Gets files with the latest date in specified folder

Args:
    folder_path_name (str): Directory path of folder storing data
    file_regex (str): Regex expression to specify file name pattern to extract file_date
    date_regex (str): Regex expression to specify date pattern in file_name
    
Returns:
    date string and file object corresponding to the file with the latest date
"""
def get_latest(folder_path_name, file_regex, date_regex):
    latest_date = dt.datetime(1000, 1, 1) # Used for comparison
    latest_date_str = "" # Represents date as formatted in file name
    latest_file = None
    
    pattern = re.compile(file_regex)
    
    for file in os.scandir(folder_path_name):
        if file.is_file():
            if not file.name.startswith('.') and pattern.match(file.name) and get_date(file.name, file_regex, date_regex, "dt") > latest_date:
                #print(file.name)
                #print(get_date(file.name, file_regex, date_regex, "dt"))
                latest_date = get_date(file.name, file_regex, date_regex, "dt")
                latest_date_str = get_date(file.name, file_regex, date_regex, "str")
                latest_file = file
        
    return latest_date_str, latest_file


# ===== Reading roster requirements from .csv files ===== #
"""Reads input file either by extracting latest file from specified folder or by user input

Args:
    test_flag (bool): Flag to trigger whether to read file without requiring user input
    folder_name (str): Directory path of folder storing data
    file_regex (str): Regex expression to specify file name pattern to extract file_date
    date_regex (str): Regex expression to specify date pattern in file_name
    
Returns:
    date string and file object corresponding to the specified file
"""
def read_file(test_flag, folder_name, file_regex, date_regex):

    if (test_flag):
        folder_path = os.path.join(os.getcwd(), folder_name)
        print("Attempting to find latest file in folder...")
        latest_date_str, latest_file = get_latest(folder_path, file_regex, date_regex)
        
        print("Latest file found: " + latest_date_str)
        input_file = pd.read_csv(latest_file)
        roster_period = latest_date_str
    else:
        # TODO: Input validation of file name and roster period
        # TODO: Update file name regex and file date regex
        
        req_folder_name = input("Input folder name: ") # Local variable
        req_folder_path = os.path.join(os.getcwd(), req_folder_name)
        
        print("Attempting to find latest file in folder...")
        latest_date_str, latest_file = get_latest(req_folder_path, file_regex, date_regex)
        
        print("Latest file found: " + latest_date_str)
        if (input("Ok to proceed? Y/N\n") == "N"):
            try:
                req_file_name = input("Please input correct file name: ") + ".csv"
                req_file_path = os.path.join(req_folder_path, req_file_name)
                input_file = pd.read_csv(req_file_path)
                roster_period = input("Please input roster period: MON-YYYY")
            except:
                try:
                    req_file_name = input("Unable to find file. Please input file name again: ") + ".csv"
                    req_file_path = os.path.join(req_folder_path, req_file_name)
                    input_file = pd.read_csv(req_file_path)
                    roster_period = input("Please input roster period: MON-YYYY")
                except:
                    print("Error: File not found")
        else:
            input_file = pd.read_csv(latest_file)
            roster_period = latest_date_str
        
    return input_file, roster_period


# ===== Read consultant data and registra data ===== #
input_c, roster_period_c = read_file(test_flag, req_folder_name, req_file_regex_c, req_date_regex)
input_r, roster_period_r = read_file(test_flag, req_folder_name, req_file_regex_r, req_date_regex)

if (roster_period_c != roster_period_r):
    print("Error: Different roster period found for registra and consultant file input")
    quit()
else:
    roster_period = roster_period_c
    # Set start and end date
    start_datetime = datetime.strptime(roster_period, '%b %Y') #Changed date regex from %b-%Y
    num_days = calendar.monthrange(start_datetime.year, start_datetime.month)[1]
    end_datetime = start_datetime + timedelta(days=num_days-1)

    # Public holidays in Singapore
    sg_holidays = []
    for date in holidays.Singapore(years=start_datetime.year).items():
        sg_holidays.append(date[0])

    # Create dataframe summarizing date details for selected month
    # Weekdays - Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3, Friday: 4
    # Weekends - Saturday: 5, Sunday: 6
    date_info = pd.DataFrame()
    date_info['date'] = pd.date_range(datetime.date(start_datetime), datetime.date(end_datetime))
    date_info['is_ph'] = [1 if val in sg_holidays else 0 for val in date_info['date']]
    date_info['is_weekday'] = [1 if val.weekday()<=4 else 0 for val in date_info['date']]
    date_info['is_weekend'] = [1 if val.weekday()>4 else 0 for val in date_info['date']]
    # To exclude days which are is_ph from is_weekday and is_weekend
    date_info['is_weekday'] = date_info.apply(lambda x: x['is_weekday'] if x['is_ph']==0 else 0, axis=1)
    date_info['is_weekend'] = date_info.apply(lambda x: x['is_weekend'] if x['is_ph']==0 else 0, axis=1)

    # Date info data - Public holiday, Weekday, Weekend
    ph_data = date_info['is_ph'].to_numpy()
    wd_data = date_info['is_weekday'].to_numpy()
    we_data = date_info['is_weekend'].to_numpy()

# ===== Read archive to extract monthly distribution score ===== #
if (test_flag):
    archive_folder_name = "Sample_Files"
else:
    archive_folder_name = input("Input archive folder name: ")
archive_folder_path = os.path.join(os.getcwd(), archive_folder_name)
archive_file_path = os.path.join(archive_folder_path, "Roster_Archive.xlsx")
roster_archive_c = pd.read_excel(archive_file_path, sheet_name='Consultant', header=[0,1], index_col=[0])
roster_archive_r = pd.read_excel(archive_file_path, sheet_name='Reg', header=[0,1], index_col=[0])

# Remove the Special_Days and Normal_Days values to keep only Total_CC
# Special_Days and Normal_Days columns are kept in case use case changes to only equalize special_days
roster_archive_c.drop('Special_Days', axis=1, level=1, inplace=True)
roster_archive_c.drop('Normal_Days', axis=1, level=1, inplace=True)
roster_archive_c = roster_archive_c.droplevel('Name', axis=1)
roster_archive_r.drop('Special_Days', axis=1, level=1, inplace=True)
roster_archive_r.drop('Normal_Days', axis=1, level=1, inplace=True)
roster_archive_r = roster_archive_r.droplevel('Name', axis=1)

# Compute min, max, mode for each month and append to roster archive dataframe
# For Roster Archive - Consultants
roster_archive_c_min_max_mode = roster_archive_c.agg(['min','max',lambda x: pd.Series.mode(x)[0]])
roster_archive_c_min_max_mode = roster_archive_c_min_max_mode.rename(index={'<lambda>': 'mode'})
roster_archive_c = roster_archive_c.append(roster_archive_c_min_max_mode)
# For Roster Archive - Regs
roster_archive_r_min_max_mode = roster_archive_r.agg(['min','max',lambda x: pd.Series.mode(x)[0]])
roster_archive_r_min_max_mode = roster_archive_r_min_max_mode.rename(index={'<lambda>': 'mode'})
roster_archive_r = roster_archive_r.append(roster_archive_r_min_max_mode)

# TODO: Re-think how to recompute carry_over 
# Min does not work: Employee will only accumulate points as cc - min >= 0 and unable to cancel out
# Max does not work: Employee will only subtract points as cc - max <= 0 and unable to cancel out
# Mode: Need to account for multiple modes (force one choice)
# Perfect distribution: Calculate monthly 

# Compute carry-over using metrics
roster_archive_c_co = roster_archive_c.copy()
roster_archive_c_co['Carry_Over'] = 0
roster_archive_r_co = roster_archive_r.copy()
roster_archive_r_co['Carry_Over'] = 0
# Assumes both archives have same columns
for col in roster_archive_c.columns:
    # If employee cc score > mode, then result > 0, to be added to total_cc in model
    # If employee cc score < mode, then result < 0, to be deducted from total_cc in model
    roster_archive_c_co[col] = roster_archive_c[col] - roster_archive_c[col].loc['mode']
    roster_archive_c_co['Carry_Over'] += roster_archive_c_co[col]
    
    roster_archive_r_co[col] = roster_archive_r[col] - roster_archive_r[col].loc['mode']
    roster_archive_r_co['Carry_Over'] += roster_archive_r_co[col]



#%% Convert file input into model input
# ===== Input data to create model instance ===== #

# Rename key constraint columns
curr_col = ['2 Day Constraint','Excluded','If Excluded, Max Calls']
new_col = ['2D_C','exc','max_exc_calls']
cons_col_dict = dict(zip(curr_col, new_col))
input_c.rename(columns=cons_col_dict, inplace=True)
input_c.fillna('', inplace=True)
input_r.rename(columns=cons_col_dict, inplace=True)
input_r.fillna('', inplace=True)

# Set up roster list, exc_list, inc_list for consultant input data
# exc_list: List of employees to exclude from total call credit variance optimisation
# inc_list: List of employees to include in total cell credit variance optimisation
roster_list_c = input_c['Name']
exc_list_c = input_c.loc[input_c['exc']=="Yes",'Name']    
inc_list_c = roster_list_c[~roster_list_c.isin(exc_list_c)]
num_consultants = len(roster_list_c)
num_consultants_exc = len(exc_list_c)

# Set up roster list, exc_list, inc_list for registra input data
# exc_list: List of employees to exclude from total call credit variance optimisation
# inc_list: List of employees to include in total cell credit variance optimisation
roster_list_r = input_r['Name']
ac_list = input_r.loc[input_r['Title']=='AC','Name']
exc_list_r = input_r.loc[input_r['exc']=="Yes",'Name']    
inc_list_r = roster_list_r[~roster_list_r.isin(exc_list_r)]
num_regs = len(roster_list_r)
num_regs_exc = len(exc_list_r)

# Extract carry-over values for all employees on current roster
# TODO: Test if it works if archive has < or > than current roster
roster_co_c = input_c.merge(roster_archive_c_co, left_on='Name', right_index=True, how='left')
roster_co_c = roster_co_c.set_index('Name')
roster_co_c = roster_co_c['Carry_Over']

roster_co_r = input_r.merge(roster_archive_r_co, left_on='Name', right_index=True, how='left')
roster_co_r = roster_co_r.set_index('Name')
roster_co_r = roster_co_r['Carry_Over']

roster_co_df_c = pd.DataFrame(roster_co_c)
roster_co_df_c.columns = pd.MultiIndex.from_tuples([('Carry_Over','')])

roster_co_df_r = pd.DataFrame(roster_co_r)
roster_co_df_r.columns = pd.MultiIndex.from_tuples([('Carry_Over','')])


# ===== Creating arrays for set-up of model constraints ===== #
"""Converts data from input file into 1-0 arrays for blocked, requested, two_day and max_excluded constraints

Args:
    num_employees (int): No. of employees in roster
    roster_list (array): List of employee names
    input_file (dataframe): Dataframe containing all input file data
    exc_list (array): List of employee names to exclude
    inc_list (array): List of employee names to include
    
Returns:
    arrays for each constraint
"""
def create_constraint_arrays(num_employees, roster_list, input_file, exc_list, inc_list):
    
    blocked_raw = np.zeros(shape=(num_employees, num_days))
    requested_raw = np.zeros(shape=(num_employees, num_days))
    
    for emp_x in range(0, num_employees):  
        for day_x in range(0, num_days):
            col = str(day_x+1)
            blocked_raw[emp_x, day_x] = (0 if input_file.iloc[emp_x][col] in blocked_types else 1)
            requested_raw[emp_x, day_x] = (1 if input_file.iloc[emp_x][col] in requested_types else 0)
                                   
    # Employee blocked data
    blocked_data = pd.DataFrame(data = blocked_raw, columns=range(0,num_days), index=roster_list)
    blocked_data = blocked_data.stack().to_dict()
    
    # Employee requested data
    requested_data = pd.DataFrame(data = requested_raw, columns=range(0,num_days), index=roster_list)
    requested_data = requested_data.stack().to_dict()

    # Employee requested - 2 day constraint (yes/no)
    two_day_constraint_raw = input_file['2D_C'].replace({'Yes': 1, 'No': 0})
    two_day_constraint_data = pd.DataFrame(data = two_day_constraint_raw.values, columns=['2_Day_Constraint'], index=roster_list)
    
    # Max calls for excluded employee
    max_excluded_raw = input_file.loc[input_file['exc']=='Yes', 'max_exc_calls']
    max_excluded_data = pd.DataFrame(data = max_excluded_raw.values, columns=['Max_Calls_Exc'], index=exc_list)
    
    return blocked_data, requested_data, two_day_constraint_data, max_excluded_data


# Set up blocked and requested data for consultant input data
blocked_c, requested_c, two_day_constraint_c, max_excluded_c = create_constraint_arrays(num_consultants, roster_list_c, input_c, exc_list_c, inc_list_c)

# Set up blocked and requested data for registra input data
blocked_r, requested_r, two_day_constraint_r, max_excluded_r = create_constraint_arrays(num_regs, roster_list_r, input_r, exc_list_r, inc_list_r)


#%% Set up Solver and output file
# Set up pyomo solver
roster_master = pyo.SolverFactory('glpk')
roster_master.options['tmlim'] = model_time_cutoff
#roster_master.options["mipgap"] = 15

# Set up .xlsx file to write results
writer = pd.ExcelWriter(roster_period + '_Roster_10s_equalcalls.xlsx', engine='xlsxwriter')   
workbook = writer.book

# Iteration Data for comparison
iteration_worksheet_r = workbook.add_worksheet('IterationData_Reg')
iteration_worksheet_c = workbook.add_worksheet('IterationData_Consultant')
roster_worksheet = workbook.add_worksheet('Roster')

writer.sheets['IterationData_Reg'] = iteration_worksheet_r
writer.sheets['IterationData_Consultant'] = iteration_worksheet_c
writer.sheets['Roster'] = roster_worksheet

# Transposed Roster for Admin Format
transposed_roster = pd.DataFrame()
transposed_roster['Date'] = pd.date_range(datetime.date(start_datetime), datetime.date(end_datetime))
transposed_roster['Day'] = transposed_roster['Date'].dt.strftime('%a')
transposed_roster['Date'] = transposed_roster['Date'].dt.day
transposed_roster['Level 2'] = ""
transposed_roster['Level 3'] = ""
transposed_roster['C/SC'] = ""

# Save requirement table in .xlsx file
required_table_r = input_r[['Name']+[str(x+1) for x in range(num_days)]]
required_table_r = required_table_r.sort_values(by='Name')
required_table_r.to_excel(writer, sheet_name='IterationData_Reg', startrow=0, startcol=0, index=False)
required_table_c = input_c[['Name']+[str(x+1) for x in range(num_days)]]
required_table_c = required_table_c.sort_values(by='Name')
required_table_c.to_excel(writer, sheet_name='IterationData_Consultant', startrow=0, startcol=0, index=False)


#%% Define Registra model

"""Set up concrete model for Reg
Objective :
    Minimize the total variance of call credit and no. of special days
    
Variables :
    x_r (Binary) : Binary indicator of whether employee x is on call on day
    absdiff_r (Non-negative Integer): Absolute difference of call credit between employee x and y
    absdiffspecial_r (Non-negative Integer): Absolute difference of no. of special days between employee x and y
        
Parameters :
    E_r, E_ac : List of registras/associate consultants
    D : List of days in roster period
    Ex_r, In_r : List of registras to exclude/include from objective optimisation
    
    daily_quota : Daily quota for entire roster period
    ph, wd, we : Indicator for whether day is a public holiday, weekday or weekend
    blocked_r, requested_r, two_day_r, max_excluded_r (Array) : Requirement tables from data input file for registras
    carryover_r : Carry-over points to be accounted for in total_cc for this period    
    
Constraints :
    daily_calls_rule_r : Daily calls must meet daily quota requirements
    blocked_days_rule_r : Employee must not be given call on a blocked day
    requested_days_rule_r : Employee must be given call on a requested day
    two_day_rule_r : Employee must not work more than 2 days in a row (unless exempted)
    max_calls_exc_rule_r : Max no. of calls for excluded employee
    
    call_credit_pos_abs_rule_r, call_credit_neg_abs_rule_r : Linearized version of constraint of sum of absolute linear residuals for call credit
    special_pos_abs_rule_r, special_neg_abs_rule_r : Linearized version of constraint of sum of absolute linear residuals for special days
    
    min_special_call_rule_r : At least 1 special day for every employee
    
"""
model_r = pyo.ConcreteModel()

# ===== Set ===== #
model_r.E_r = pyo.Set(initialize=roster_list_r) # Employees - Regs
model_r.E_ac = pyo.Set(initialize=ac_list) # Employees - Associate Consultants
model_r.D = pyo.Set(initialize=range(num_days)) # Days
model_r.Ex_r = pyo.Set(initialize=exc_list_r) # Excluded Employees
model_r.In_r = pyo.Set(initialize=inc_list_r) # Included Employees


# ===== Parameters ===== #
# For date-related computations
model_r.dailyquota = pyo.Param(model_r.D, within=pyo.NonNegativeIntegers, default=1)
model_r.ph = pyo.Param(model_r.D, within=pyo.Binary, initialize=ph_data) #Is public holiday parameter
model_r.wd = pyo.Param(model_r.D, within=pyo.Binary, initialize=wd_data) #Is weekday parameter
model_r.we = pyo.Param(model_r.D, within=pyo.Binary, initialize=we_data) #Is weekend parameter

# For constraints - Regs
model_r.blocked_r = pyo.Param(model_r.E_r, model_r.D, within=pyo.Binary, initialize=blocked_r, default=1) 
model_r.requested_r = pyo.Param(model_r.E_r, model_r.D, within=pyo.Binary, initialize=requested_r, default=0)
model_r.two_day_r = pyo.Param(model_r.E_r, within=pyo.Binary, initialize=two_day_constraint_r, default=1)
model_r.max_excluded_r = pyo.Param(model_r.Ex_r, within=pyo.NonNegativeIntegers, initialize=max_excluded_r, default=num_days)

# For cc equalization over previous roster periods
model_r.carryover_r = pyo.Param(model_r.E_r, within=pyo.Integers, initialize=roster_co_r)

# ===== Decision Variables ===== #
# 1 if employee to be scheduled for on-call on that day, Else 0
model_r.x_r = pyo.Var(model_r.E_r, model_r.D, within=pyo.Binary)
model_r.absdiff_r = pyo.Var(model_r.In_r, model_r.In_r, within=pyo.NonNegativeIntegers) # Absolute variance in Call Credit - Regs
model_r.absdiffspecial_r = pyo.Var(model_r.In_r, model_r.In_r, within=pyo.NonNegativeIntegers) # Absolute variance in # of Special Days - Regs


# ===== Objective: ===== #
# Note: Default objective is minimization, to maximize: sense = pyo.maximize
# To minimize total call credit variance

# Calculate total call credit - Reg
def call_credit_total_rule_r(model, i):
    return sum(model.x_r[i,j]*model.ph[j]*3 + model.x_r[i,j]*model.we[j]*3 + model.x_r[i,j]*model.wd[j]*2 for j in model.D) + model.carryover_r[i]
model_r.cctotal_r = pyo.Expression(model_r.E_r, rule=call_credit_total_rule_r)

# Calculate difference in call credit between each employee - Reg
def call_credit_diff_rule_r(model, a, b):
    return model.cctotal_r[a] - model.cctotal_r[b]
model_r.ccdiff_r = pyo.Expression(model_r.E_r, model_r.E_r, rule=call_credit_diff_rule_r)

# Calculate total number of special days - Reg
def special_total_rule_r(model, i):
    return sum(model.x_r[i,j]*model.ph[j] + model.x_r[i,j]*model.we[j] for j in model.D)
model_r.special_r = pyo.Expression(model_r.E_r, rule=special_total_rule_r)

# Calculate total special days variance - Reg
def special_diff_rule_r(model, a, b):
    return model.special_r[a] - model.special_r[b]
model_r.specialdiff_r = pyo.Expression(model_r.E_r, model_r.E_r, rule=special_diff_rule_r)

# Calculate total variance - Reg
def total_var_rule_r(model):
    return sum(sum(model.absdiff_r[i,j] + model.absdiffspecial_r[i,j] for i in model.In_r) for j in model.In_r)
    # return sum(sum(model.absdiff_r[i,j] + model.absdiffspecial_r[i,j] for i in model.In_r) for j in model.In_r)
model_r.totvar = pyo.Objective(rule=total_var_rule_r)



# ===== Constraints ===== #
model_r.cuts = pyo.ConstraintList()

# Constraint: Daily calls must meet daily quota requirements - Reg
def daily_calls_rule_r(model, d):
    return sum(model.x_r[e,d] for e in model.E_r) == model.dailyquota[d]
model_r.dailytotal_r = pyo.Constraint(model_r.D, rule=daily_calls_rule_r)

# Constraint: Employee must not be given call on a blocked day - Reg
def blocked_days_rule_r(model, i, j):
    return model.x_r[i,j] <= model.blocked_r[i,j]
model_r.blockedtotal_r = pyo.Constraint(model_r.E_r, model_r.D, rule=blocked_days_rule_r)

# Constraint: Employee must be given call on a requested day - Reg
def requested_days_rule_r(model, i, j):
    return model.x_r[i,j] >= model.requested_r[i,j]
model_r.requestedtotal_r = pyo.Constraint(model_r.E_r, model_r.D, rule=requested_days_rule_r)

# Constraint: Employee must not work more than 2 days in a row - Reg
# Allow user to remove this constraint if necessary
def two_day_rule_r(model, i, j):
    try:
        day_plus_one = model.x_r[i,j+1]
    except:
        day_plus_one = 0
    try:
        day_plus_two = model.x_r[i,j+2]
    except:
        day_plus_two = 0
    
    if model.two_day_r[i]==1:        
        return model.x_r[i,j] + day_plus_one + day_plus_two <= 1
    else:
        return model.x_r[i,j] + day_plus_one + day_plus_two <= 3
model_r.twodaytotal_r = pyo.Constraint(model_r.E_r, model_r.D, rule=two_day_rule_r)

# Constraint: Max no. of calls for excluded employee - Reg
def max_calls_exc_rule_r(model,i):
    return sum(model.x_r[i,j] for j in model.D) == model.max_excluded_r[i]
model_r.maxcallexc_r = pyo.Constraint(model_r.Ex_r, rule=max_calls_exc_rule_r) 

# Linearized version of constraint of sum of absolute linear residuals for call credit - Reg
def call_credit_pos_abs_rule_r(model, a, b):
    return model.absdiff_r[a,b] >= model.ccdiff_r[a,b]
model_r.ccposdiff_r = pyo.Constraint(model_r.In_r, model_r.In_r, rule=call_credit_pos_abs_rule_r)

def call_credit_neg_abs_rule_r(model, a, b):
    return model.absdiff_r[a,b] >= -model.ccdiff_r[a,b]
model_r.ccnegdiff_r = pyo.Constraint(model_r.In_r, model_r.In_r, rule=call_credit_neg_abs_rule_r)

# Linearized version of constraint of sum of absolute linear residuals for special days - Reg
def special_pos_abs_rule_r(model, a, b):
    return model.absdiffspecial_r[a,b] >= model.specialdiff_r[a,b]
model_r.specialposdiff_r = pyo.Constraint(model_r.In_r, model_r.In_r, rule=special_pos_abs_rule_r)

def special_neg_abs_rule_r(model, a, b):
    return model.absdiffspecial_r[a,b] >= -model.specialdiff_r[a,b]
model_r.specialnegdiff_r = pyo.Constraint(model_r.In_r, model_r.In_r, rule=special_neg_abs_rule_r)

# Constraint: At least 1 special day for every employee - Reg
def min_special_call_rule_r(model,i):
    return sum(model.x_r[i,j]*model.we[j] + model.x_r[i,j]*model.ph[j] for j in model.D) >= 1
model_r.minspec_r = pyo.Constraint(model_r.E_r, rule=min_special_call_rule_r)



#%% Define Consultant model

"""Set up concrete model for Consultant
Objective :
    Minimize the total variance of call credit and no. of special days
    
Variables :
    x_c (Binary) : Binary indicator of whether employee x is on call on day
    absdiff_c (Non-negative Integer): Absolute difference of call credit between employee x and y
    absdiffspecial_c (Non-negative Integer): Absolute difference of no. of special days between employee x and y
        
Parameters :
    E_c: List of onsultants
    D : List of days in roster period
    Ex_c, In_c : List of consultants to exclude/include from objective optimisation
    
    daily_quota : Daily quota for entire roster period
    ph, wd, we : Indicator for whether day is a public holiday, weekday or weekend
    blocked_c, requested_c, two_day_c, max_excluded_c (Array) : Requirement tables from data input file for consultants
    carryover_c : Carry-over points to be accounted for in total_cc for this period    
    
Constraints :
    daily_calls_rule_c : Daily calls must meet daily quota requirements
    blocked_days_rule_c : Employee must not be given call on a blocked day
    requested_days_rule_c : Employee must be given call on a requested day
    two_day_rule_c : Employee must not work more than 2 days in a row (unless exempted)
    max_calls_exc_rule_c : Max no. of calls for excluded employee
    
    call_credit_pos_abs_rule_c, call_credit_neg_abs_rule_c : Linearized version of constraint of sum of absolute linear residuals for call credit (only includes inc_list)
    special_pos_abs_rule_c, special_neg_abs_rule_c : Linearized version of constraint of sum of absolute linear residuals for special days (only includes inc_list)
    
    min_special_call_rule_c : At least 1 special day for every employee (only includes inc_list)
    min_ac_call_rule_c : At least 1 call with AC for every employee (all employees)
    
"""
model_c = pyo.ConcreteModel() 

# ===== Set ===== #
model_c.E_c = pyo.Set(initialize=roster_list_c) # Employees - Consultants
model_c.D = pyo.Set(initialize=range(num_days)) # Days
model_c.Ex_c = pyo.Set(initialize=exc_list_c) # Excluded Employees
model_c.In_c = pyo.Set(initialize=inc_list_c) # Included Employees


# ===== Parameters ===== #
# For date-related computations
model_c.dailyquota = pyo.Param(model_c.D, within=pyo.NonNegativeIntegers, default=1)
model_c.ph = pyo.Param(model_c.D, within=pyo.Binary, initialize=ph_data) #Is public holiday parameter
model_c.wd = pyo.Param(model_c.D, within=pyo.Binary, initialize=wd_data) #Is weekday parameter
model_c.we = pyo.Param(model_c.D, within=pyo.Binary, initialize=we_data) #Is weekend parameter

# For constraints - Consultants
model_c.blocked_c = pyo.Param(model_c.E_c, model_c.D, within=pyo.Binary, initialize=blocked_c, default=1) 
model_c.requested_c = pyo.Param(model_c.E_c, model_c.D, within=pyo.Binary, initialize=requested_c, default=0)
model_c.two_day_c = pyo.Param(model_c.E_c, within=pyo.Binary, initialize=two_day_constraint_c, default=1)
model_c.max_excluded_c = pyo.Param(model_c.Ex_c, within=pyo.NonNegativeIntegers, initialize=max_excluded_c, default=num_days)

# For cc equalization over previous roster periods
model_c.carryover_c = pyo.Param(model_c.E_c, within=pyo.Integers, initialize=roster_co_c)


# ===== Decision Variables ===== #
# 1 if employee to be scheduled for on-call on that day, Else 0
model_c.x_c = pyo.Var(model_c.E_c, model_c.D, within=pyo.Binary)
model_c.absdiff_c = pyo.Var(model_c.In_c, model_c.In_c, within=pyo.NonNegativeIntegers) # Absolute variance in Call Credit - Consultant
model_c.absdiffspecial_c = pyo.Var(model_c.In_c, model_c.In_c, within=pyo.NonNegativeIntegers) # Absolute variance in # of Special Days - Consultant


# ===== Objective: ===== #
# Note: Default objective is minimization, to maximize: sense = pyo.maximize
# To minimize total call credit variance

# Calculate total call credit - Consultant
def call_credit_total_rule_c(model, i):
    return sum(model.x_c[i,j]*model.ph[j]*3 + model.x_c[i,j]*model.we[j]*3 + model.x_c[i,j]*model.wd[j]*2 for j in model.D) + model.carryover_c[i]
model_c.cctotal_c = pyo.Expression(model_c.E_c, rule=call_credit_total_rule_c)

# Calculate difference in call credit between each employee - Consultant
def call_credit_diff_rule_c(model, a, b):
    return model.cctotal_c[a] - model.cctotal_c[b]
model_c.ccdiff_c = pyo.Expression(model_c.E_c, model_c.E_c, rule=call_credit_diff_rule_c)

# Calculate total number of special days - Consultant
def special_total_rule_c(model, i):
    return sum(model.x_c[i,j]*model.ph[j] + model.x_c[i,j]*model.we[j] for j in model.D)
model_c.special_c = pyo.Expression(model_c.E_c, rule=special_total_rule_c)

# Calculate total special days variance - Consultant
def special_diff_rule_c(model, a, b):
    return model.special_c[a] - model.special_c[b]
model_c.specialdiff_c = pyo.Expression(model_c.E_c, model_c.E_c, rule=special_diff_rule_c)

# Calculate total variance
def total_var_rule_c(model):
    return sum(sum(model.absdiff_c[i,j] + model.absdiffspecial_c[i,j] for i in model.In_c) for j in model.In_c)
model_c.totvar = pyo.Objective(rule=total_var_rule_c)


# ===== Constraints ===== #
model_c.cuts = pyo.ConstraintList()

# Constraint: Daily calls must meet daily quota requirements - Consultant
def daily_calls_rule_c(model, d):
    return sum(model.x_c[e,d] for e in model.E_c) == model.dailyquota[d]
model_c.dailytotal_c = pyo.Constraint(model_c.D, rule=daily_calls_rule_c)

# Constraint: Employee must not be given call on a blocked day - Consultant
def blocked_days_rule_c(model, i, j):
    return model.x_c[i,j] <= model.blocked_c[i,j]
model_c.blockedtotal_c = pyo.Constraint(model_c.E_c, model_c.D, rule=blocked_days_rule_c)

# Constraint: Employee must be given call on a requested day - Consultant
def requested_days_rule_c(model, i, j):
    return model.x_c[i,j] >= model.requested_c[i,j]
model_c.requestedtotal_c = pyo.Constraint(model_c.E_c, model_c.D, rule=requested_days_rule_c)

# Constraint: Employee must not work more than 2 days in a row - Consultant
# Allow user to remove this constraint if necessary
def two_day_rule_c(model, i, j):
    try:
        day_plus_one = model.x_c[i,j+1]
    except:
        day_plus_one = 0
    try:
        day_plus_two = model.x_c[i,j+2]
    except:
        day_plus_two = 0
    
    if model.two_day_c[i]==1:        
        return model.x_c[i,j] + day_plus_one + day_plus_two <= 1
    else:
        return model.x_c[i,j] + day_plus_one + day_plus_two <= 3
model_c.twodaytotal_c = pyo.Constraint(model_c.E_c, model_c.D, rule=two_day_rule_c)

# Constraint: Max no. of calls for excluded employee - Consultant
def max_calls_exc_rule_c(model,i):
    return sum(model.x_c[i,j] for j in model.D) == model.max_excluded_c[i]
model_c.maxcallexc_c = pyo.Constraint(model_c.Ex_c, rule=max_calls_exc_rule_c) 

# Linearized version of constraint of sum of absolute linear residuals for call credit - Consultant
def call_credit_pos_abs_rule_c(model, a, b):
    return model.absdiff_c[a,b] >= model.ccdiff_c[a,b]
model_c.ccposdiff_c = pyo.Constraint(model_c.In_c, model_c.In_c, rule=call_credit_pos_abs_rule_c)

def call_credit_neg_abs_rule_c(model, a, b):
    return model.absdiff_c[a,b] >= -model.ccdiff_c[a,b]
model_c.ccnegdiff_c = pyo.Constraint(model_c.In_c, model_c.In_c, rule=call_credit_neg_abs_rule_c)

# Linearized version of constraint of sum of absolute linear residuals for special days - Consultant
def special_pos_abs_rule_c(model, a, b):
    return model.absdiffspecial_c[a,b] >= model.specialdiff_c[a,b]
model_c.specialposdiff_c = pyo.Constraint(model_c.In_c, model_c.In_c, rule=special_pos_abs_rule_c)

def special_neg_abs_rule_c(model, a, b):
    return model.absdiffspecial_c[a,b] >= -model.specialdiff_c[a,b]
model_c.specialnegdiff_c = pyo.Constraint(model_c.In_c, model_c.In_c, rule=special_neg_abs_rule_c)

# Constraint: At least 1 special day for every employee - Consultant
def min_special_call_rule_c(model,i):
    return sum(model.x_c[i,j]*model.we[j] + model.x_c[i,j]*model.ph[j] for j in model.D) >= 1
model_c.minspec_c = pyo.Constraint(model_c.E_c, rule=min_special_call_rule_c)


#%% Run model

"""Returns initials from name string

Args:
    fullname (string) : Full employee name

Returns:
    initials (string) : String of initials from employee name

"""
def get_initials(fullname):
    xs = (fullname)
    name_list = xs.split()
    first = name_list[0][0]
    second = name_list[1][0]

    return(first.upper() + second.upper())

"""Gets variable outputs from model solve

Args:
    instance (model) : Optimisation model
    model_iteration (int) : Iteration of model run
    input_type (int) : Type of data input
    roster_list (array) : List of employees
    inc_list (array) : List of employees to be included in total variance computation
    exc_list (array) : List of employees to be excluded from total variance computation
    ac_list (array) : List of associate consultants (applicable only for reg output)
    
Returns:
    result_table (array) : Array of AC output (days where an AC is on call)
    result_schedule (dataframe) : Dataframe of result output including computed columns
    transposed_schedule (dataframe) : Dataframe of result output transposed
    result_row (array) : Array of AC output (name of AC on call)
"""
def get_results(instance, model_iteration, input_type, roster_list, inc_list, exc_list, ac_list, ac_result, transposed_roster):
    
    # Extract variable output in 2D array format
    # Source: https://stackoverflow.com/questions/67491499/how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
    output_vars = instance.component_map(ctype=pyo.Var)
    var_series = []   # Collection to hold the converted variables
    for k in output_vars.keys():   # Map of {name:pyo.Var}
        
        # Create pd.Series for each variable
        v = output_vars[k]
        s = pd.Series(v.extract_values(), index=v.extract_values().keys()) 

        # Unstack in case of multi-indexed series
        if type(s.index[0]) == tuple: # Series is multi-indexed
            s = s.unstack(level=1)
        else:
            s = pd.DataFrame(s) # Force transition from Series -> df
            #print(s)

        # Multi-index the columns
        s.columns = pd.MultiIndex.from_tuples([(k, t) for t in s.columns])
        var_series.append(s)
        result_schedule = pd.concat(var_series, axis=1)
                 
    # To specify which variables references to use to update table
    if (input_type == "Reg"):
        x_var_name = "x_r"
        absdiff_name = "absdiff_r"
        absdiffspecial_name = "absdiffspecial_r"
    elif (input_type == "Consultant"):
        x_var_name = "x_c"
        absdiff_name = "absdiff_c"
        absdiffspecial_name = "absdiffspecial_c"
           
        
    # Create array of result output for use for Reg input - only include AC employees
    result_table = np.zeros(shape=num_days)
    result_row = pd.DataFrame(index=[0], columns=range(0, num_days))
    if (input_type == "Reg"):
        for ac_x in ac_list:
            for day_x in range(0, num_days):
                if (result_schedule.loc[ac_x, (x_var_name, day_x)]==1):
                    result_table[day_x] = 1
                    result_row[day_x].loc[0] = get_initials(ac_x)
    
    # Create dataframe for result schedule for print output #
    # Compute the no. of normal days and special days and total call credit using variable output
    result_schedule['Normal_Days'] = 0
    result_schedule['Special_Days'] = 0
    result_schedule['Total_CC'] = 0
    for day_x in range(0, num_days):
        if (ph_data[day_x] == 1) or (we_data[day_x] == 1):
            result_schedule['Special_Days'] += result_schedule[x_var_name, day_x]
        if (wd_data[day_x] == 1):
            result_schedule['Normal_Days'] += result_schedule[x_var_name, day_x]
    result_schedule['Total_CC'] += result_schedule['Special_Days']*3 + result_schedule['Normal_Days']*2
    
    # Output carry over from previous roster months for both Consultant input and Reg input    
    if (input_type == "Reg"):
        result_schedule = result_schedule.merge(pd.DataFrame(roster_co_df_r), left_index=True, right_index=True)
    # Output final num of AC per Consultant for Consultant input
    elif (input_type == "Consultant"):
        result_schedule = result_schedule.merge(pd.DataFrame(roster_co_df_c), left_index=True, right_index=True)
        result_schedule['Num_AC'] = sum(result_schedule[x_var_name, day_x]*ac_result[day_x] for day_x in range(0, num_days))
    
    # Compute absolute difference in call credit and no. of special days
    col_dict = {}
    keys = [x for x in range(0, num_days)]
    values = [x for x in range(1, num_days+1)]
    for i in keys:
        col_dict[i] = values[i]
    result_schedule = result_schedule.rename(columns=col_dict)
    result_schedule = result_schedule.rename(columns={x_var_name: "Model Iteration " + str(model_iteration), absdiff_name: "CC_Diff", absdiffspecial_name: "Special_Diff"})
    for i in roster_list:
        for j in inc_list:
            result_schedule.loc[i,('CC_Diff',j)] = abs(result_schedule.loc[i,('Total_CC','')] - result_schedule.loc[j,('Total_CC','')])
            result_schedule.loc[i,('Special_Diff',j)] = abs(result_schedule.loc[i,('Special_Days','')] - result_schedule.loc[j,('Special_Days','')])

    # Generate transposed roster - Fill on-call person for that day's roster 
    if (input_type == "Reg"):
        transposed_roster['Level 2'] = ""
        transposed_roster['Level 3'] = ""
        for day_x in range(0, num_days):
            for employee_x in roster_list:
                if (result_schedule.loc[employee_x, ('Model Iteration ' + str(model_iteration), day_x+1)]==1):
                    if employee_x in ac_list.values:
                        transposed_roster.loc[day_x, 'Level 3'] = employee_x
                    else:
                        transposed_roster.loc[day_x, 'Level 2'] = employee_x
    elif (input_type == "Consultant"):
        for day_x in range(0, num_days):
            for employee_x in roster_list:
                if (result_schedule.loc[employee_x, ('Model Iteration ' + str(model_iteration), day_x+1)]==1):
                    transposed_roster.loc[day_x, 'C/SC'] = employee_x
                           
    return result_table, result_schedule, transposed_roster, result_row


"""Saves result_schedule in output .xlsx file

Args:
    model_iteration (int) : Model Iteration
    result_r, result_c (dataframe) : Dataframe of result schedule
    result_ac_row (array) : Dataframe of AC on call
    num_regs, num_consultants (int) : No. of employess (regs/consultants respectively)
    iteration_data_row_r, iteration_data_row_c (int) : Row to start printing result schedule
    transposed_roster (dataframe) : Dataframe of transposed roster
    roster_data_col (int) : Column to start printing transposed roster schedule
        
Returns:
    iteration_data_row_r, iteration_data_row_c and roster_data_col
"""
def print_results(model_iteration, 
                 result_r, num_regs, iteration_data_row_r, result_ac_row,
                 result_c, num_consultants, iteration_data_row_c,
                 transposed_roster, roster_data_col):
    
    print("\n==== Iteration", model_iteration)
    print(result_r)
    print(result_c)
    
    if (model_iteration != 1):
        # 2 - Column for Day of month and Day of Week, 3 - Columns for Level 2,3,C/SC and 1 - Blank column
        roster_data_col += 2 + 3 + 1 
        # 3 - Header rows, 1 - Blank row, 2 - AC Summary Row
        iteration_data_row_r += 3 + num_regs + 1
        iteration_data_row_c += 3 + num_consultants + 1 + 1 # Add one for AC call roster
    
    result_r.to_excel(writer, sheet_name="IterationData_Reg", startrow=iteration_data_row_r, startcol=0)
    result_c.to_excel(writer, sheet_name='IterationData_Consultant', startrow=iteration_data_row_c, startcol=0)
    
    result_ac = pd.DataFrame(result_ac_row)
    result_ac.to_excel(writer, sheet_name='IterationData_Consultant', startrow=(iteration_data_row_c + 3 + num_consultants), startcol=1, header=False, index=False)
    
    transposed_roster = transposed_roster.set_index(['Date'])
    transposed_roster.columns = pd.MultiIndex.from_product([['Model Iteration ' + str(model_iteration)], list(transposed_roster.columns)])
    transposed_roster.to_excel(writer, sheet_name='Roster', startrow=0, startcol=roster_data_col)

    return iteration_data_row_r, iteration_data_row_c, roster_data_col

 
# Run registra model for first time
model_iteration = 1
results_r = roster_master.solve(model_r) #, tee=True)

if (results_r.solver.termination_condition == pyo.TerminationCondition.infeasible):
    print('Error: Infeasible Registra Model')
else:
    # Extract data for ac_consultants
    result_table_r, result_schedule_r, transposed_roster, result_row_r = get_results(model_r, model_iteration, "Reg", roster_list_r, inc_list_r, exc_list_r, ac_list, None, transposed_roster)
    
    # Add constraint to consultant model based on registra solved solution
    # Constraint: Consultant must have a call with at least one AC
    model_c.accall = pyo.Param(model_c.D, within=pyo.Binary, initialize=result_table_r, mutable=True)
    def min_ac_call_rule_c(model, e):
        return sum(model.x_c[e,d]*model.accall[d] for d in model.D) >= 1
    model_c.total_ac = pyo.Constraint(model_c.E_c, rule=min_ac_call_rule_c)
    
    # Run consultant model
    results_c = roster_master.solve(model_c) #, tee=True)
    result_table_c, result_schedule_c, transposed_roster, result_row_c = get_results(model_c, model_iteration, "Consultant", roster_list_c, inc_list_c, exc_list_c, ac_list, result_table_r, transposed_roster)
    
    # Run registra model until a feasible solution is found for consultant model
    while (results_c.solver.termination_condition == pyo.TerminationCondition.infeasible):
        # Add infeasible registra solution cut to exclude previously found solution from next run
        # Source: https://pyomo.readthedocs.io/en/stable/working_models.html    
        expr_r = 0
        for e in model_r.E_r:
            for d in model_r.D:
                if pyo.value(model_r.x_r[e,d]) < 0.5:
                    expr_r += model_r.x_r[e,d]
                else:
                    expr_r += (1 - model_r.x_r[e,d])
        model_r.cuts.add(expr_r>=1)
        results_r = roster_master.solve(model_r) #, tee=True)
        result_table_r, result_schedule_r, transposed_roster, result_row_r = get_results(model_r, model_iteration, "Reg", roster_list_r, inc_list_r, exc_list_r, ac_list, None, transposed_roster)
        
        # Update constraint to contain new result output
        for d in range(num_days):
            model_c.accall[d] = result_table_r[d]
        # model_c.acccall.display()
        
        # Run consultant model with new solution from registra model as min_ac_constraint
        results_c = roster_master.solve(model_c) #, tee=True)
        result_table_c, result_schedule_c, transposed_roster, result_row_c = get_results(model_c, model_iteration, "Consultant", roster_list_c, inc_list_c, exc_list_c, ac_list, result_table_r, transposed_roster)
    
    
    iteration_data_row_r, iteration_data_row_c, roster_data_col = print_results(model_iteration, 
                                                                                result_schedule_r, num_regs, num_regs + 2, result_row_r,
                                                                                result_schedule_c, num_consultants, num_consultants + 2,
                                                                                transposed_roster, 0)

# ===== Iterate ===== #
# Add cut to exclude previously found solution
# Source: https://pyomo.readthedocs.io/en/stable/working_models.html
for i in range(model_iterations-1):
    model_iteration += 1
    
    # Add previous solution to exclude from registra model
    expr_r = 0
    for e in model_r.E_r:
        for d in model_r.D:
            if pyo.value(model_r.x_r[e,d])<0.5:
                expr_r += model_r.x_r[e,d]
            else: 
                expr_r += (1-model_r.x_r[e,d])
    model_r.cuts.add(expr_r>=1)
    
    # Add previous solution to exclude from consultant model
    expr_c = 0
    for e in model_c.E_c:
        for d in model_c.D:
            if pyo.value(model_c.x_c[e,d])<0.5:
                expr_c += model_c.x_c[e,d]
            else:
                expr_c += (1-model_c.x_c[e,d])
    model_c.cuts.add(expr_c>=1)
    
    # Find next solution for both registra and consultant model
    results_r = roster_master.solve(model_r) #, tee=True)

    if (results_r.solver.termination_condition == pyo.TerminationCondition.infeasible):
        print('Error: Infeasible Registra Model')
    else:
        # Extract data for ac_consultants
        result_table_r, result_schedule_r, transposed_roster, result_row_r = get_results(model_r, model_iteration, "Reg", roster_list_r, inc_list_r, exc_list_r, ac_list, None, transposed_roster)
        
        # Add constraint to consultant model based on registra solved solution
        # Constraint: Consultant must have a call with at least one AC
        for d in range(num_days):
            model_c.accall[d] = result_table_r[d]
        # model_c.accall.display()
        
        # Run consultant model
        results_c = roster_master.solve(model_c) #, tee=True)
        result_table_c, result_schedule_c, transposed_roster, result_row_c = get_results(model_c, model_iteration, "Consultant", roster_list_c, inc_list_c, exc_list_c, ac_list, result_table_r, transposed_roster)
        
        # Run registra model until a feasible solution is found for consultant model
        while (results_c.solver.termination_condition == pyo.TerminationCondition.infeasible):
            # Add infeasible registra solution cut to exclude previously found solution from next run
            # Source: https://pyomo.readthedocs.io/en/stable/working_models.html    
            expr_r = 0
            for e in model_r.E_r:
                for d in model_r.D:
                    if pyo.value(model_r.x_r[e,d]) < 0.5:
                        expr_r += model_r.x_r[e,d]
                    else:
                        expr_r += (1 - model_r.x_r[e,d])
            model_r.cuts.add(expr_r>=1)
            results_r = roster_master.solve(model_r) #, tee=True)
            result_table_r, result_schedule_r, transposed_roster, result_row_r = get_results(model_r, model_iteration, "Reg", roster_list_r, inc_list_r, exc_list_r, ac_list, None, transposed_roster)
            
            # Update constraint to contain new result output
            for d in range(num_days):
                model_c.accall[d] = result_table_r[d]
            # model_c.accall.display()
            
            # Run consultant model with new solution from registra model as min_ac_constraint
            results_c = roster_master.solve(model_c) #, tee=True)
            result_table_c, result_schedule_c, transposed_roster, result_row_c = get_results(model_c, model_iteration, "Consultant", roster_list_c, inc_list_c, exc_list_c, ac_list, result_table_r, transposed_roster)
        
        
        iteration_data_row_r, iteration_data_row_c, roster_data_col = print_results(model_iteration, 
                                                                                    result_schedule_r, num_regs, iteration_data_row_r, result_row_r,
                                                                                    result_schedule_c, num_consultants, iteration_data_row_c,
                                                                                    transposed_roster, roster_data_col)


#%% Set output file format

# Set all basic format types
# Light red fill
cell_format_red = workbook.add_format({'bg_color': '#FFC7CE', 'border': 1})
# Light yellow fill with dark yellow text
cell_format_yellow = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500', 'border': 1})
# Light green fill
cell_format_green = workbook.add_format({'bg_color': '#C6EFCE', 'border': 1})
# Bold font
cell_format_bold = workbook.add_format({'bold': 1})

# 28 days: Col AC, 29 days: Col AD, 30 days: Col AE, 31 days: Col AF
"""Returns col reference for day of month input

Args:
    argument (int) : Day of month
        
Returns:
    Column reference as string object
"""
def switch_end_col(argument):
    switcher = {
        1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
        6: "G", 7: "H", 8: "I", 9: "J", 10: "K",
        11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
        16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
        21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z",
        26: "AA", 27: "AB", 28: "AC", 29: "AD", 30: "AE",
        31: "AF"
    }
    return switcher.get(argument, "Invalid days")

# Generate col reference for each date
date_info['day'] = date_info['date'].dt.day
date_info['col_ref'] = date_info.apply(lambda row: switch_end_col(row['day']), axis=1)
special_days = date_info[(date_info['is_weekend']==1) | (date_info['is_ph']==1)]

# ===== Define Criteria Formula ===== #
# Criteria formula for conditional formatting is automatically generated from defined blocked and requested types
# Blocked criteria
blocked_criteria = '=OR('
for x in blocked_types:
    blocked_criteria = blocked_criteria + 'B2="' + x + '",'
blocked_criteria = blocked_criteria[:-1]
blocked_criteria = blocked_criteria + ")"

# Requested criteria
requested_criteria = '=OR('
for x in requested_types:
    requested_criteria = requested_criteria + 'B2="' + x + '",'
requested_criteria = requested_criteria[:-1]
requested_criteria = requested_criteria + ")"


"""Set format for iteration_worksheet tabs in .xlsx file

Args:
    num_employees (int) : No. of employees 
    iteration_worksheet (worksheet) : Worksheet to update - Consultant or Reg
    is_consultant (bool) : Setting format for consultant sheet
"""
def set_format(num_employees, iteration_worksheet, is_consultant):
    
    # Cell references for iteration worksheet
    day_row = num_employees + 2 + 2
    start_row = num_employees + 2 + 4
    start_col = "B"
    end_row = start_row + num_employees - 1
    end_col = switch_end_col(num_days)
    req_table_ref = start_col + str(2) + ":" + end_col + str(num_employees+1)
    
    # Set for required table - different worksheets for different input_type
    for special_x in special_days.index:
        day_cell_ref = special_days['col_ref'].loc[special_x] + str(1)
        iteration_worksheet.write(day_cell_ref, str(special_days['day'].loc[special_x]), cell_format_yellow)
        iteration_worksheet.conditional_format(req_table_ref,
                                           {'type': 'formula',
                                            'criteria': blocked_criteria,
                                            'format': cell_format_red})
        iteration_worksheet.conditional_format(req_table_ref,
                                           {'type': 'formula',
                                            'criteria': requested_criteria,
                                            'format': cell_format_green})
        
    # Set conditional formatting for each iteration table for easy checking
    for iteration_x in range(model_iterations):
        
        # For each model iteration - there are 4 + num_employee rows to format
        # Row 2: Days of the month - add yellow highlight for special days
        # Row 4 to (3 + num_employees) - add green highlight for requested, red for blocked
        
        table_ref = start_col + str(start_row) + ":" + end_col + str(end_row)
        for special_x in special_days.index:
            day_cell_ref = date_info['col_ref'].iloc[special_x] + str(day_row)
            iteration_worksheet.write(day_cell_ref, date_info['day'].iloc[special_x], cell_format_yellow)

        iteration_worksheet.conditional_format(table_ref,
                                               {'type': 'formula',
                                               'criteria': blocked_criteria,
                                               'format': cell_format_red})
        iteration_worksheet.conditional_format(table_ref,
                                               {'type': 'formula',
                                               'criteria': requested_criteria,
                                               'format': cell_format_green})
        iteration_worksheet.conditional_format(table_ref, 
                                               {'type': 'cell',
                                                'criteria': '==',
                                                'value': 1,
                                                'format': cell_format_bold})
        
        day_row += num_employees + 4
        start_row += num_employees + 4
        end_row += num_employees + 4
        
        # Add additional row to account for additional row for AC roster
        if (is_consultant):
            day_row += 1
            start_row += 1
            end_row += 1
            

    iteration_worksheet.set_column(0, 0, 16)
    iteration_worksheet.set_column(1, num_days, 3)
    iteration_worksheet.set_column(num_days+1, num_days+1+num_employees, 16)
    iteration_worksheet.set_column(num_days+1+num_employees+1, num_days+1+num_employees+3, 15)  
    iteration_worksheet.set_column(num_days+1+num_employees+3, num_days+1+num_employees+5, 12) 
   
# Set format for iteration worksheet
set_format(num_regs, iteration_worksheet_r, False)
set_format(num_consultants, iteration_worksheet_c, True)

# Set format for roster worksheet
for model_i in range(model_iterations):
    col_i = model_i*6
    
    roster_worksheet.set_column(col_i, col_i, 4)
    roster_worksheet.set_column(col_i+1, col_i+1, 4)
    roster_worksheet.set_column(col_i+2, col_i+4, 16)
    roster_worksheet.set_column(col_i+5, col_i+5, 2)
    

"""Returns col reference for day of month input

Args:
    num_days (int) : Number of days in month
    num_inc (int) : Number of employees diff col is calculated for
        
Returns:
    Column reference of difference columns (CC_Diff, Special_Diff)
"""
def get_diff_col_ref(num_days, num_inc):
    # Note: col_idx is 0-based, 28 is referencing col for day 27
    col_idx_ref_mapping = {
        28: "AC", 29: "AD", 30: "AE", 31: "AF", 32: "AG", 
        33: "AH", 34: "AI", 35: "AJ", 36: "AK", 37: "AL",
        38: "AM", 39: "AN", 40: "AO", 41: "AP", 42: "AQ",
        43: "AR", 44: "AS", 45: "AT", 46: "AU", 47: "AV",
        48: "AW", 49: "AX", 50: "AY", 51: "AZ", 52: "BA",
        53: "BB", 54: "BC", 55: "BD", 56: "BE", 57: "BF",
        58: "BG", 59: "BH", 60: "BI"
    }
    
    start_col_idx = num_days + 1 # Note: First Name column alr included in num_days
    start_col_ref = col_idx_ref_mapping.get(start_col_idx, "Invalid days")
    end_col_idx = start_col_idx + num_inc + num_inc - 1
    end_col_ref = col_idx_ref_mapping.get(end_col_idx, "Invalid days")
    
    return start_col_ref + ":" + end_col_ref
    
# Hide CC_Diff, Special_Diff columns
#num_inc_r = inc_list_r.size
#diff_col_ref_r = get_diff_col_ref(num_days, num_inc_r)
#iteration_worksheet_r.set_column(diff_col_ref_r, None, None, {'hidden': True})

#num_inc_c = inc_list_c.size
#diff_col_ref_c = get_diff_col_ref(num_days, num_inc_c)
#iteration_worksheet_c.set_column(diff_col_ref_c, None, None, {'hidden': True})

# Print carry-over data for reference
roster_archive_r


writer.close()



            
            
            
            
            
            
            
            
            
            
            
