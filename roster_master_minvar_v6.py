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


        
#%% Reading data

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


# ===== Set programme run settings ===== #
test_flag = True
model_time_cutoff = 5
model_iterations = 10

req_folder_name = "Sample_Files"
month_regex = "((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) 20[2-9][0-9])"
req_file_regex_c = "Call requests \(Consultant\) - " + month_regex + ".csv"
req_file_regex_r = "Call requests \(Reg\) - " + month_regex + ".csv"
req_date_regex = "%b %Y"

# Read consultant data
input_file, roster_period = read_file(test_flag, req_folder_name, req_file_regex_c, req_date_regex)


# TODO: Add reading of previous month's roster schedules

# ===== Reading previous months' roster schedules from .xlsx file ===== #
# Each tab has roster schedule for each month
# Each row in schedule specifies employee info





#%% Converting file input into model input
# ===== Input data to create model instance ===== #

# Rename key constraint columns
curr_col = ['2 Day Constraint','Excluded','If Excluded, Max Calls']
new_col = ['2D_C','exc','max_exc_calls']
con_col_dict = dict(zip(curr_col, new_col))
input_file.rename(columns=con_col_dict, inplace=True)
input_file.fillna('', inplace=True)

# Set up roster list, exc_list, inc_list
# exc_list: List of employees to exclude from total call credit variance optimisation
# inc_list: List of employees to include in total cell credit variance optimisation
roster_list = input_file['Name']
exc_list = input_file.loc[input_file['exc']=="Yes",'Name']    
inc_list = roster_list[~roster_list.isin(exc_list)]
num_employees = len(roster_list)
num_exc = len(exc_list)

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

# Legend for Block Types: AL,TL,X,NSL,OIL,ML,PL,FCL,CCL,KKH,PDL
# Legend for Request: R, CR
blocked_types = ['AL','TL','X','NSL','OIL','ML','PL','FCL','CCL','KKH','PDL']
requested_types = ['R','CR']

blocked_raw = np.zeros(shape=(num_employees, num_days))
requested_raw = np.empty(shape=(num_employees, num_days))

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



# Date info data - Public holiday, Weekday, Weekend
ph_data = date_info['is_ph'].to_numpy()
wd_data = date_info['is_weekday'].to_numpy()
we_data = date_info['is_weekend'].to_numpy()





# ===== Create model ===== #
roster_master = pyo.SolverFactory('glpk')
#roster_master.options["mipgap"] = 15
roster_master.options['tmlim'] = model_time_cutoff


# Set up concrete model
model = pyo.ConcreteModel()

# == Set == #
# Employees
model.E = pyo.Set(initialize=roster_list)
# Days
model.D = pyo.Set(initialize=range(num_days))
# Excluded Employees
model.Ex = pyo.Set(initialize=exc_list)
# Included Employees
model.In = pyo.Set(initialize=inc_list)

# == Parameters == #
model.dailyquota = pyo.Param(model.D, within=pyo.NonNegativeIntegers, default=1)
model.ph = pyo.Param(model.D, within=pyo.Binary, initialize=ph_data) #Is public holiday parameter
model.wd = pyo.Param(model.D, within=pyo.Binary, initialize=wd_data) #Is weekday parameter
model.we = pyo.Param(model.D, within=pyo.Binary, initialize=we_data) #Is weekend parameter
model.blocked = pyo.Param(model.E, model.D, within=pyo.Binary, initialize=blocked_data, default=1) 
model.requested = pyo.Param(model.E, model.D, within=pyo.Binary, initialize=requested_data, default=0)
model.two_day = pyo.Param(model.E, within=pyo.Binary, initialize=two_day_constraint_data, default=1)
model.max_excluded = pyo.Param(model.Ex, within=pyo.NonNegativeIntegers, initialize=max_excluded_data, default=num_days)

# == Decision Variables == #
# 1 if employee to be scheduled for on-call on that day, Else 0
model.x = pyo.Var(model.E, model.D, within=pyo.Binary)
model.absdiff = pyo.Var(model.In, model.In, within=pyo.NonNegativeIntegers) # Absolute variance in Call Credit
model.absdiffspecial = pyo.Var(model.In, model.In, within=pyo.NonNegativeIntegers) # Absolute variance in # of Special Days

# == Objective: ==
# Note: Default objective is minimization, to maximize: sense = pyo.maximize
# To minimize total call credit variance

# Calculate total call credit
def call_credit_total_rule(model, i):
    return sum(model.x[i,j]*model.ph[j]*3 + model.x[i,j]*model.we[j]*3 + model.x[i,j]*model.wd[j]*2 for j in model.D)
model.cctotal = pyo.Expression(model.E, rule=call_credit_total_rule)

# Calculate difference in call credit between each employee
def call_credit_diff_rule(model, a, b):
    return model.cctotal[a] - model.cctotal[b]
model.ccdiff = pyo.Expression(model.E, model.E, rule=call_credit_diff_rule)

# Calculate total number of special days
def special_total_rule(model, i):
    return sum(model.x[i,j]*model.ph[j] + model.x[i,j]*model.we[j] for j in model.D)
model.special = pyo.Expression(model.E, rule=special_total_rule)

# Calculate total special days variance
def special_diff_rule(model, a, b):
    return model.special[a] - model.special[b]
model.specialdiff = pyo.Expression(model.E, model.E, rule=special_diff_rule)

# Calculate total variance
def total_var_rule(model):
    return sum(sum(model.absdiff[i,j] + model.absdiffspecial[i,j] for i in model.In) for j in model.In)
model.totvar = pyo.Objective(rule=total_var_rule)

# == Constraints == #
model.cuts = pyo.ConstraintList()

# Constraint: Daily calls must meet daily quota requirements
def daily_calls_rule(model, d):
    return sum(model.x[e,d] for e in model.E) == model.dailyquota[d]
model.dailytotal = pyo.Constraint(model.D, rule=daily_calls_rule)

# Constraint: Employee must not be given call on a blocked day
def blocked_days_rule(model, i, j):
    return model.x[i,j] <= model.blocked[i,j]
model.blockedtotal = pyo.Constraint(model.E, model.D, rule=blocked_days_rule)

# Constraint: Employee must be given call on a requested day
def requested_days_rule(model, i, j):
    return model.x[i,j] >= model.requested[i,j]
model.requestedtotal = pyo.Constraint(model.E, model.D, rule=requested_days_rule)

# Constraint: Employee must not work more than 2 days in a row
# Allow user to remove this constraint if necessary
def two_day_rule(model, i, j):
    try:
        day_plus_one = model.x[i,j+1]
    except:
        day_plus_one = 0
    try:
        day_plus_two = model.x[i,j+2]
    except:
        day_plus_two = 0
    
    if model.two_day[i]==1:        
        return model.x[i,j] + day_plus_one + day_plus_two <= 1
    else:
        return model.x[i,j] + day_plus_one + day_plus_two <= 3
model.twodaytotal = pyo.Constraint(model.E, model.D, rule=two_day_rule)

# Constraint: Max no. of calls for excluded employee
def max_calls_exc_rule(model,i):
    return sum(model.x[i,j] for j in model.D) <= model.max_excluded[i]
model.maxcallexc = pyo.Constraint(model.Ex, rule=max_calls_exc_rule)

# Linearized version of constraint of sum of absolute linear residuals for call credit
def call_credit_pos_abs_rule(model, a, b):
    return model.absdiff[a,b] >= model.ccdiff[a,b]
model.ccposdiff = pyo.Constraint(model.In, model.In, rule=call_credit_pos_abs_rule)

def call_credit_neg_abs_rule(model, a, b):
    return model.absdiff[a,b] >= -model.ccdiff[a,b]
model.ccnegdiff = pyo.Constraint(model.In, model.In, rule=call_credit_neg_abs_rule)

# Linearized version of constraint of sum of absolute linear residuals for special days
def special_pos_abs_rule(model, a, b):
    return model.absdiffspecial[a,b] >= model.specialdiff[a,b]
model.specialposdiff = pyo.Constraint(model.In, model.In, rule=special_pos_abs_rule)

def special_neg_abs_rule(model, a, b):
    return model.absdiffspecial[a,b] >= -model.specialdiff[a,b]
model.specialnegdiff = pyo.Constraint(model.In, model.In, rule=special_neg_abs_rule)

# Constraint: At least 1 special day for every employee
def min_special_call_rule(model,i):
    return sum(model.x[i,j]*model.we[j] + model.x[i,j]*model.ph[j] for j in model.D) >= 1
model.minspec = pyo.Constraint(model.In, rule=min_special_call_rule)


# ===== Run model ===== #
writer = pd.ExcelWriter(roster_period + '_Roster.xlsx', engine='xlsxwriter')   
workbook = writer.book
iteration_worksheet = workbook.add_worksheet('Iteration Data')
roster_worksheet = workbook.add_worksheet('Roster')
writer.sheets['Iteration Data'] = iteration_worksheet
writer.sheets['Roster'] =roster_worksheet

# Transposed Roster for Admin Format
transposed_roster = pd.DataFrame()
transposed_roster['Date'] = pd.date_range(datetime.date(start_datetime), datetime.date(end_datetime))
transposed_roster['Day'] = transposed_roster['Date'].dt.strftime('%a')
transposed_roster['Date'] = transposed_roster['Date'].dt.day
transposed_roster['On-Call Team'] = ""


#model.display()

results = roster_master.solve(model, tee=True)
results.write()
model_iteration = 1
iteration_data_row = 0
roster_data_col = 0


required_table = input_file[['Name']+[str(x+1) for x in range(num_days)]]
required_table = required_table.sort_values(by='Name')
required_table.to_excel(writer, sheet_name='Iteration Data', startrow=iteration_data_row, startcol=0, index=False)   
iteration_data_row += num_employees + 2

# ===== Print model solution ===== #
def print_variables(instance, format_type=2):
    
    # Format: In single column
    if (format_type==1):
        result_data = {(i,j,v.name): pyo.value(v) for (i,j), v in model.x.items()}
        result_schedule = pd.DataFrame.from_dict(result_data, orient='index', columns=["variable value"])
    
    # Format: In 2D array
    # Source: https://stackoverflow.com/questions/67491499/how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
    if (format_type==2):
        output_vars = model.component_map(ctype=pyo.Var)
        var_series = []   # Collection to hold the converted variables
        for k in output_vars.keys():   # Map of {name:pyo.Var}
            v = output_vars[k]

            # Create pd.Series for each 
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
         
        #Exclude for now as using sample data of 6 days not full month    
        #result_schedule.columns = pd.date_range(datetime.date(start_datetime), datetime.date(end_datetime))
        
        result_schedule['Normal_Days'] = 0
        result_schedule['Special_Days'] = 0
        for day_x in range(0, num_days):
            if (ph_data[day_x] == 1) or (we_data[day_x] == 1):
                result_schedule['Special_Days'] += result_schedule['x', day_x]
            if (wd_data[day_x] == 1):
                result_schedule['Normal_Days'] += result_schedule['x', day_x]
        result_schedule['Total_CC'] = result_schedule['Special_Days']*3 + result_schedule['Normal_Days']*2

        col_dict = {}
        keys = [x for x in range(0, num_days)]
        values = [x for x in range(1, num_days+1)]
        for i in keys:
            col_dict[i] = values[i]
        result_schedule = result_schedule.rename(columns=col_dict)
        result_schedule = result_schedule.rename(columns={'x': "Model Iteration " + str(model_iteration), "absdiff": "Absolute_CC_Diff"})
        for i in roster_list:
            for j in inc_list:
                result_schedule['Absolute_CC_Diff', j].loc[i] = abs(result_schedule['Total_CC'].loc[i] - result_schedule['Total_CC'].loc[j])
        

        #Fill on-call person for that day's roster
        for day_x in range(0, num_days):
            for employee_x in roster_list:
                if (result_schedule.loc[employee_x,('Model Iteration ' + str(model_iteration), day_x+1)]==1):
                    transposed_roster.loc[day_x, 'On-Call Team'] = employee_x
                    
        result_schedule.to_excel(writer, sheet_name='Iteration Data', startrow=iteration_data_row, startcol=0)   
        transposed_roster.rename(columns={'On-Call Team': 'Iteration ' + str(model_iteration)}, inplace=True)
        transposed_roster.to_excel(writer, sheet_name='Roster', startrow=0, startcol=roster_data_col, index=False)
        transposed_roster.rename(columns={'Iteration ' + str(model_iteration): 'On-Call Team'}, inplace=True)
        
        return result_schedule

roster = print_variables(model, 2)
print("\n==== Iteration", model_iteration)
print(roster)
model_iteration += 1
iteration_data_row += 3 + num_employees + 1
roster_data_col += 4




# ===== Iterate ===== #
# Add cut to exclude previously found solution
# Source: https://pyomo.readthedocs.io/en/stable/working_models.html
for i in range(model_iterations-1):
    expr = 0
    for e in model.E:
        for d in model.D:
            if pyo.value(model.x[e,d])<0.5:
                expr += model.x[e,d]
            else: 
                expr += (1-model.x[e,d])
    model.cuts.add(expr>=1)
    results = roster_master.solve(model, tee=True)
    print("\n==== Iteration", model_iteration)
    roster = print_variables(model, 2)
    print(roster)
    model_iteration += 1
    iteration_data_row += 3 + num_employees + 1
    roster_data_col += 4
    #model.cctotal.display()
    #model.ccvar.display()
    #model.display()
                      
    
# ===== Set Excel Format ===== #
# Light red fill
cell_format_red = workbook.add_format({'bg_color': '#FFC7CE', 'border': 1})
# Light yellow fill with dark yellow text
cell_format_yellow = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500', 'border': 1})
# Light green fill
cell_format_green = workbook.add_format({'bg_color': '#C6EFCE', 'border': 1})
# Bold font
cell_format_bold = workbook.add_format({'bold': 1})

# Legend for Block Types: AL,TL,X,NSL,OIL,ML,PL,FCL,CCL
# Legend for Request: R
# 28 days: Col AC, 29 days: Col AD, 30 days: Col AE, 31 days: Col AF
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

day_row = num_employees + 2 + 2
start_row = num_employees + 2 + 4
end_row = start_row + num_employees - 1
start_col = "B"
end_col = switch_end_col(num_days)

date_info['day'] = date_info['date'].dt.day
date_info['col_ref'] = date_info.apply(lambda row: switch_end_col(row['day']), axis=1)
special_days = date_info[(date_info['is_weekend']==1) | (date_info['is_ph']==1)]


# Set for required table
for special_x in special_days.index:
    day_cell_ref = special_days['col_ref'].loc[special_x] + str(1)
    iteration_worksheet.write(day_cell_ref, str(special_days['day'].loc[special_x]), cell_format_yellow)
req_table_ref = start_col + str(2) + ":" + end_col + str(num_employees+1)

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
    
    day_row_ref = start_col + str(day_row) + ":" + end_col + str(day_row)
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

iteration_worksheet.set_column(0, 0, 15)
iteration_worksheet.set_column(1, num_days, 3)
iteration_worksheet.set_column(num_days+1, num_days+1+num_employees, 15)
iteration_worksheet.set_column(num_days+1+num_employees+1, num_days+1+num_employees+3, 15)  
iteration_worksheet.set_column(num_days+1+num_employees+3, num_days+1+num_employees+5, 10)  


# For each model iteration - there are 4 + num_employee rows to format
# Row 2: Days of the month - add yellow highlight for special days
# Row 4 to (3 + num_employees) - add green highlight for requested, red for blocked
for model_i in range(model_iterations):
    col_i = model_i*4
    
    roster_worksheet.set_column(col_i, col_i, 4)
    roster_worksheet.set_column(col_i+1, col_i+1, 4)
    roster_worksheet.set_column(col_i+2, col_i+2, 15)
    roster_worksheet.set_column(col_i+3, col_i+3, 2)    

writer.close()  

   



            
            
            
            
            
            
            
            
            
            
            
