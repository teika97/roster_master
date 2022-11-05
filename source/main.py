import os
import re
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import calendar
import datetime as dt
from datetime import datetime, timedelta
from storage import Setting, Month, Holiday, Archive, Request, InputData
from tkinter import *
from tkinter.messagebox import showinfo
from tkinter import ttk
import reader

cwd = os.getcwd()

# Initialize all variables
model_settings = Setting()
sg_holidays = Holiday()
roster_month = Month()
input_r, archive_r, request_r = InputData(), Archive(), Request()
input_c, archive_c, request_c = InputData(), Archive(), Request()

root = Tk()
root.title("Roster Master")
mainframe = ttk.Frame(root, padding = "3 3 12 12")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Validate the input for selected month/tab - must be in %b %Y date format
# Show error message when invalid input
def validate():
    date_pattern = model_settings.month_regex
    if re.search(date_pattern, selected_value.get()) is None:
        showinfo(
            title='Input Error',
            message=f'Please input month as Mon YYYY format'
        )
        return False
    return True

def hide(event):
    if (input_type.get() == "Gsheet - Selected Tab") | (input_type.get() == "CSV - Selected Month"):
        selected_input.pack()
    else:
        selected_input.pack_forget()

def extract_data():
    """ handle the input changed event """

    # Interface will be drop down list and not allow for error in user input
    emp_type = employee_type.get()
    input_dict = {"Gsheet - First Tab":'1', "CSV - Latest Month":'2', "CSV - Selected Month":'3', "Gsheet - Selected Tab":'4'}
    input_idx = input_dict[input_type.get()]
    selected_val = selected_value.get()

    if (emp_type == 'Both') | (emp_type == 'Reg') | (emp_type == 'Con'):
        if (emp_type == 'Both') | (emp_type == 'Reg'):
            file_date_str, raw_input_r, roster_archive_r = reader.read_data(cwd, model_settings, "Reg", input_idx, selected_val) 
            input_r.set_inputdata(file_date_str, raw_input_r, roster_archive_r)
        if (emp_type == 'Both') | (emp_type == 'Con'):
            file_date_str, raw_input_c, roster_archive_c = reader.read_data(cwd, model_settings, "Con", input_idx, selected_val) 
            input_c.set_inputdata(file_date_str, raw_input_c, roster_archive_c)

    else:
        showinfo(
            title='Input Error!',
            message=f'Invalid value for employee_type'
        )

    roster_period = file_date_str
    sg_holidays.set_holidays(roster_period)
    roster_month.set_month(roster_period, sg_holidays.sg_holidays)

    if (emp_type == 'Both') | (emp_type == 'Reg'):
        archive_r.set_archive('Registra', roster_archive_r)
        request_r.set_request(roster_month, 'Registra', raw_input_r, False, False, archive_r.carryover_data)
        request_r.print_details()

    if (emp_type == 'Both') | (emp_type == 'Con'):
        archive_c.set_archive('Consultant', roster_archive_c)
        request_c.set_request(roster_month, 'Registra', raw_input_c, False, False, archive_c.carryover_data)
        request_c.print_details()
        

button = Button(root, text="Extract Data", command=extract_data)
button.pack()

# Set up drop down list for employee_type
employee_type = StringVar()
employee_type.set('Both') # Default value
both_employee_type = ttk.Radiobutton(root, text="Both", variable=employee_type, value='Both')
reg_employee_type = ttk.Radiobutton(root, text="Reg Only", variable=employee_type, value='Reg')
con_employee_type = ttk.Radiobutton(root, text="Consultant Only", variable=employee_type, value='Con')
both_employee_type.pack(fill=X, padx=5, pady=5)
reg_employee_type.pack(fill=X, padx=5, pady=5)
con_employee_type.pack(fill=X, padx=5, pady=5)

# Set up drop down list for input_type
input_type = StringVar()
input_type.set("Gsheet - First Tab") # Default value
input_type_dropdown = ttk.Combobox(root, textvariable=input_type)
input_type_dropdown['values'] = ("Gsheet - First Tab", "CSV - Latest Month", "CSV - Selected Month", "Gsheet - Selected Tab")
input_type_dropdown.state(["readonly"])
input_type_dropdown.pack(fill=X, padx=5, pady=5)
input_type_dropdown.bind('<<ComboboxSelected>>', hide)

selected_value = StringVar()
first_day_of_next_month = (dt.datetime.now().replace(day=1) + dt.timedelta(days=32)).replace(day=1)
selected_value.set(first_day_of_next_month.strftime('%b %Y'))
selected_input = Entry(root, textvariable=selected_value)
selected_input.config(validate='focusout', validatecommand=validate)


root.mainloop()



# Get user input on whether to run for
# (i) Both registra and consultant model
# (ii) Only registra model
# (iii) Only consultant model
# TODO: Allow for updating of default option
#employee_type = input("\nDefault employee type is 'Both Reg and Consultant'\nKeep ('Y') or change ('1': Reg Only, '2': Con Only)\n?: ")

# Get user input on whether to get model input for
# (i) Latest month file from default "data" folder if no input - Only csv supported for now
# (ii) Selected month from default "data" folder - Only csv supported for now
# (iii) First tab from gsheet
# (iv) Specific tab from gsheet
# TODO: Allow for updating of default option
# For now no support for selecting of folder - can add function to change default folder
#input_type = input("\nDefault import type is 'Gsheet - First Tab'\nKeep ('Y') or change ('1': CSV - Latest Month, '2': CSV - Selected Month, '3': Gsheet - Selected Tab)\n?: ")
#selected_value = None
#if (input_type == '2'):
#    selected_value = input("\nSelected Month (as Mon YYYY): ")
#elif (input_type == '3'):
#    selected_value = input("\nSelected Tab Name (as Mon YYYY): ")

# Interface will be drop down list and not allow for error in user input
#selected_value = None
#if (employee_type == 'Both') | (employee_type == 'Reg') | (employee_type == 'Con'):
#    if (employee_type == 'Both') | (employee_type == 'Reg'):
#        file_date_str, input_r, roster_archive_r = reader.read_data(cwd, model_settings, "Reg", input_type, selected_value) 
#    if (employee_type == 'Both') | (employee_type == 'Con'):
#        file_date_str, input_c, roster_archive_c = reader.read_data(cwd, model_settings, "Con", input_type, selected_value) 
#else:
#    raise Exception("Input error: Invalid value for employee_type")

#roster_period = file_date_str
#sg_holidays = Holiday(roster_period)
#roster_month = Month(roster_period, sg_holidays.sg_holidays)

#if (employee_type == 'Both') | (employee_type == 'Reg'):
#    archive_r = Archive('Registra', roster_archive_r)
#    request_r = Request(roster_month, 'Registra', input_r, False, False, archive_r.carryover_data)
#    request_r.print_details()
#if (employee_type == 'Both') | (employee_type == 'Con'):
#    archive_c = Archive('Consultant', roster_archive_c)
#    request_c = Request(roster_month, 'Consultant', input_c, False, False, archive_c.carryover_data)
#    request_c.print_details()
