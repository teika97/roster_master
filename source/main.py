import os
import re
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import calendar
import datetime as dt
from datetime import datetime, timedelta
from storage import Setting, Month, Holiday, Archive, Request
from tkinter import *
from tkinter import ttk
import reader

cwd = os.getcwd()

model_settings = Setting()

# Get user input on whether to run for
# (i) Both registra and consultant model
# (ii) Only registra model
# (iii) Only consultant model
# TODO: Allow for updating of default option
employee_type = input("\nDefault employee type is 'Both Reg and Consultant'\nKeep ('Y') or change ('1': Reg Only, '2': Con Only)\n?: ")

# Get user input on whether to get model input for
# (i) Latest month file from default "data" folder if no input - Only csv supported for now
# (ii) Selected month from default "data" folder - Only csv supported for now
# (iii) First tab from gsheet
# (iv) Specific tab from gsheet
# TODO: Allow for updating of default option
# For now no support for selecting of folder - can add function to change default folder
input_type = input("\nDefault import type is 'Gsheet - First Tab'\nKeep ('Y') or change ('1': CSV - Latest Month, '2': CSV - Selected Month, '3': Gsheet - Selected Tab)\n?: ")
selected_value = None
if (input_type == '2'):
    selected_value = input("\nSelected Month (as Mon YYYY): ")
elif (input_type == '3'):
    selected_value = input("\nSelected Tab Name (as Mon YYYY): ")

# Interface will be drop down list and not allow for error in user input
if (employee_type == 'Y') | (employee_type == '1') | (employee_type == '2'):
    if (employee_type == 'Y') | (employee_type == '1'):
        file_date_str, input_r, roster_archive_r = reader.read_data(cwd, model_settings, "Reg", input_type, selected_value) 
    if (employee_type == 'Y') | (employee_type == '2'):
        file_date_str, input_c, roster_archive_c = reader.read_data(cwd, model_settings, "Con", input_type, selected_value) 
else:
    raise Exception("Input error: Invalid value for employee_type")

roster_period = file_date_str
sg_holidays = Holiday(roster_period)
roster_month = Month(roster_period, sg_holidays.sg_holidays)

if (employee_type == 'Y') | (employee_type == '1'):
    archive_r = Archive('Registra', roster_archive_r)
    request_r = Request(roster_month, 'Registra', input_r, False, False, archive_r.carryover_data)
    request_r.print_details()
if (employee_type == 'Y') | (employee_type == '2'):
    archive_c = Archive('Consultant', roster_archive_c)
    request_c = Request(roster_month, 'Consultant', input_c, False, False, archive_c.carryover_data)
    request_c.print_details()
