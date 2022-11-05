# Classes for data entities related to (i) Request (ii) Archive (iii) Month data (iv) Schedule
import re
import pandas as pd
import numpy as np
import calendar
import datetime as dt
from datetime import datetime, timedelta
import holidays

BLOCKED_TYPES = ['AL','TL','X','NSL','OIL','ML','PL','FCL','CCL','KKH','PDL','ACLS','HL','Hero']
REQUESTED_TYPES = ['R','CR']


class Setting:
    def __init__(self, req_folder_name = "data", model_time_cutoff = 10, model_iterations = 30, is_maxcalls = True, use_cheat = False):
        self.month_regex = "((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) 20[2-9][0-9])"
        self.req_file_prefix_c = "Call requests (Consultant) - "
        self.req_file_prefix_r = "Call requests (Reg) - "
        self.req_file_regex_c = "Call requests \(Consultant\) - "+ self.month_regex + ".csv"
        self.req_file_regex_r = "Call requests \(Reg\) - "+ self.month_regex + ".csv"
        
        self.req_folder_name = req_folder_name
        self.req_date_regex = "%b %Y"
        self.roster_archive_file_name = ("Roster_Archive_cheat.xlsx" if use_cheat else "Roster_Archive.xlsx")  

        self.passcode_json_file_name = 'passcode_key.json'

        self.model_time_cutoff = model_time_cutoff
        self.model_iterations = model_iterations
        self.is_maxcalls = is_maxcalls # Sets up maxcall constraint as max calls or equal calls
        self.use_cheat = use_cheat

    # Depending on setting of the is_maxcalls field - sets the output file name accordingly
    def set_output_file_name(self):
        self.output_file_name = "_Roster_" + str(model_time_cutoff) + ("s_maxcalls.xlsx" if is_maxcalls else "s_equalcalls.xlsx")


class Holiday:
    def __init__(self):
        self.sg_holidays = []

    def set_holidays(self, roster_period):
        # Public holidays in Singapore
        for date in holidays.Singapore(years=datetime.strptime(roster_period, '%b %Y').year).items():
            self.sg_holidays.append(date[0])



# Create dataframe summarizing date details for selected month
class Month:
    def __init__(self):
        self.month_data = pd.DataFrame()

    def set_month(self, roster_period, holidays):
        start_datetime = datetime.strptime(roster_period, '%b %Y') #Changed date regex from %b-%Y

        self.roster_period = roster_period
        self.num_days = calendar.monthrange(start_datetime.year, start_datetime.month)[1]
        end_datetime = start_datetime + timedelta(days=self.num_days-1)

        # Weekdays - Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3, Friday: 4
        # Weekends - Saturday: 5, Sunday: 6
        self.month_data['date'] = pd.date_range(start_datetime.date(), end_datetime.date())
        self.month_data['day'] = pd.to_datetime(self.month_data['date']).dt.day
        self.month_data['date'] = pd.to_datetime(self.month_data['date']).dt.date
        self.month_data['is_ph'] = [1 if val in holidays else 0 for val in self.month_data['date']]
        self.month_data['is_weekday'] = [1 if val.weekday()<=4 else 0 for val in self.month_data['date']]
        self.month_data['is_weekend'] = [1 if val.weekday()>4 else 0 for val in self.month_data['date']]
        # To exclude days which are is_ph from is_weekday and is_weekend
        self.month_data['is_weekday'] = self.month_data.apply(lambda x: x['is_weekday'] if x['is_ph']==0 else 0, axis=1)
        self.month_data['is_weekend'] = self.month_data.apply(lambda x: x['is_weekend'] if x['is_ph']==0 else 0, axis=1)
        
    def ph_data(self):
        return self.month_data['is_ph'].to_numpy()

    def wd_data(self):
        return self.month_data['is_weekday'].to_numpy()

    def we_data(self):
        return self.month_data['is_weekend'].to_numpy()

class InputData:
    def set_inputdata(self, file_date_str, input_data, roster_archive):
        self.file_date_str = file_date_str
        self.input_data = input_data
        self.roster_archive = roster_archive

class Archive:
    def set_archive(self, employee_type, raw_data):
        self.employee_type = employee_type
        self.roster_list = raw_data.index

        raw_data.drop('Special_Days', axis=1, level=1, inplace=True)
        raw_data.drop('Normal_Days', axis=1, level=1, inplace=True)
        raw_data = raw_data.droplevel('Name', axis=1)
        min_max = raw_data.agg(['min','max'])
        mode = pd.DataFrame(raw_data.agg(lambda x: pd.Series.mode(x)[0])).transpose().rename({0: 'mode'})

        archive_data = pd.concat([raw_data, min_max, mode])
        carryover_data = archive_data.copy()
        carryover_data['Carry_Over'] = 0
        for col in archive_data.columns:
            # If employee cc score > mode, then result > 0, to be added to total_cc in model
            # If employee cc score < mode, then result < 0, to be deducted from total_cc in model
            carryover_data[col] = archive_data[col] - archive_data[col].loc['mode']
            carryover_data['Carry_Over'] += carryover_data[col]
            carryover_data['Carry_Over'] = carryover_data['Carry_Over'].fillna(0)

        self.archive_data = archive_data
        self.carryover_data = carryover_data

class Request:
    def set_request(self, month, employee_type, raw_data, override_twoday, is_maxcalls, carryover_data):
        self.month = month
        self.employee_type = employee_type
        self.raw_data = raw_data.fillna('')

        self.roster_list = self.raw_data['Name']
        self.exc_list = self.raw_data.loc[self.raw_data['Exc']=='Yes','Name']
        self.inc_list = self.roster_list.loc[~self.roster_list.isin(self.exc_list)]
        self.num_employees = len(self.roster_list)
        self.num_employees_exc = len(self.exc_list)

        if (employee_type == 'Registra'):
            self.ac_list = self.raw_data.loc[self.raw_data['Title']=='AC', 'Name']

        blocked_raw = np.zeros(shape=(self.num_employees, self.month.num_days))
        requested_raw = np.zeros(shape=(self.num_employees, self.month.num_days))
    
        for emp_x in range(0, self.num_employees):  
            for day_x in range(0, self.month.num_days):
                col = str(day_x+1)

                #Add 0 in front of all days before day 10
                if (day_x<9):
                    if (col not in raw_data.columns):
                        col = "0" + str(day_x+1)

                blocked_raw[emp_x, day_x] = (0 if self.raw_data.iloc[emp_x][col] in BLOCKED_TYPES else 1)
                requested_raw[emp_x, day_x] = (1 if self.raw_data.iloc[emp_x][col] in REQUESTED_TYPES else 0)                

        # Employee blocked data
        blocked_data = pd.DataFrame(data = blocked_raw, columns = range(0, self.month.num_days), index = self.roster_list)
        self.blocked_data = blocked_data.stack().to_dict()
    
        # Employee requested data
        requested_data = pd.DataFrame(data = requested_raw, columns = range(0, self.month.num_days), index = self.roster_list)
        self.requested_data = requested_data.stack().to_dict()
        
        # Employee requested - 2 day constraint (yes/no)
        if (override_twoday):
            two_day_constraint_raw = self.raw_data['2D'].replace({'Yes': 0, 'No': 0}) # Set No for all employees to override original file input
        else:  
            two_day_constraint_raw = self.raw_data['2D'].replace({'Yes': 1, 'No': 0})
        two_day_constraint_data = pd.DataFrame(data = two_day_constraint_raw.values, columns = ['2_Day_Constraint'], index = self.roster_list)
        self.twoday_data = two_day_constraint_data

        # Max calls for excluded employee
        max_excluded_raw = self.raw_data.loc[self.raw_data['Exc']=='Yes', 'Max']
        max_excluded_data = pd.DataFrame(data = max_excluded_raw.values, columns=['Max_Calls_Exc'], index=self.exc_list)
        self.max_data = max_excluded_data

        # Merge with archive carry over data
        # Extract carry-over values for all employees on current roster
        carryover_data = raw_data.merge(carryover_data, left_on='Name', right_index=True, how='left')
        carryover_data = carryover_data.set_index('Name')
        carryover_data = carryover_data['Carry_Over']
        self.carryover_data = carryover_data

    def print_details(self):
        
        print("Roster period: " + self.month.roster_period + ", Employee type: " + self.employee_type)
        
        print("Raw data: ")
        print(self.raw_data)

        print("Roster list: ")
        print(self.roster_list)

        print("Excluded: ")
        print(self.exc_list)

        print("Included: ")
        print(self.inc_list)

        print("Two day data: ")
        print(self.twoday_data)

        print("Max data: ")
        print(self.max_data)

        print("Carry over data: ")
        print(self.carryover_data)


    # TODO: Add methods to check for the feasibility of the request for:
    # (i) Daily feasibility: i.e. the day has at least one person
    # (ii) For each range of 3 days there is enough people to cover all days
    # (iii) More than one employee requests for a particular day
    # (iv) Not enough blank days for the at least 1 special day requirement to be fulfilled
    # Else returns false with the exact dates that require amendment for model to work

