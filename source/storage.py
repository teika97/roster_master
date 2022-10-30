import pandas as pd
import numpy as np
import calendar
import datetime as dt
from datetime import datetime, timedelta

BLOCKED_TYPES = ['AL','TL','X','NSL','OIL','ML','PL','FCL','CCL','KKH','PDL','ACLS','HL','Hero']
REQUESTED_TYPES = ['R','CR']

# Create dataframe summarizing date details for selected month
class Month:
    def __init__(self, roster_period, holidays):
        
        start_datetime = datetime.strptime(roster_period, '%b %Y') #Changed date regex from %b-%Y

        self.roster_period = roster_period
        self.num_days = calendar.monthrange(start_datetime.year, start_datetime.month)[1]
        end_datetime = start_datetime + timedelta(days=num_days-1)

        # Weekdays - Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3, Friday: 4
        # Weekends - Saturday: 5, Sunday: 6
        month_data = pd.DataFrame()
        month_data['date'] = pd.date_range(start_datetime.date(), end_datetime.date())
        month_data['day'] = pd.to_datetime(month_data['date']).dt.day
        month_data['date'] = pd.to_datetime(month_data['date']).dt.date
        month_data['is_ph'] = [1 if val in holidays else 0 for val in month_data['date']]
        month_data['is_weekday'] = [1 if val.weekday()<=4 else 0 for val in month_data['date']]
        month_data['is_weekend'] = [1 if val.weekday()>4 else 0 for val in month_data['date']]
        # To exclude days which are is_ph from is_weekday and is_weekend
        month_data['is_weekday'] = month_data.apply(lambda x: x['is_weekday'] if x['is_ph']==0 else 0, axis=1)
        month_data['is_weekend'] = month_data.apply(lambda x: x['is_weekend'] if x['is_ph']==0 else 0, axis=1)

        self.month_data = month_data
        
    def ph_data(self):
        return self.month_data['is_ph'].to_numpy()

    def wd_data(self):
        return self.month_data['is_weekday'].to_numpy()

    def we_data(self):
        return self.month_data['is_weekend'].to_numpy()


class Archive:
    def __init__(self, employee_type, raw_data):
        self.employee_type = employee_type
        self.roster_list = self.raw_data['Name']

        raw_data.drop('Special_Days', axis=1, level=1, inplace=True)
        raw_data.drop('Normal_Days', axis=1, level=1, inplace=True)
        raw_data = raw_data.droplevel('Name', axis=1)

        self.min_max_mode = raw_data.agg(['min','max',lambda x: pd.Series.mode(x)[0]])
        self.min_max_mode = self.min_max_mode.rename(index={'<lambda>': 'mode'})
        
        archive_data = pd.concat([raw_data, self.min_max_mode])
        carryover_data = archive_data.copy()
        carryover_data['Carry_Over'] = 0
        for col in archive_data.columns:
            # If employee cc score > mode, then result > 0, to be added to total_cc in model
            # If employee cc score < mode, then result < 0, to be deducted from total_cc in model
            carryover_data[col] = archive_data[col] - archive_data[col].loc['mode']
            carryover_data['Carry_Over'] += archive_data[col]
            carryover_data['Carry_Over'] = carryover_data['Carry_Over'].fillna(0)
        carryover_data['Name'] = self.roster_list

        self.archive_data = archive_data
        self.carryover_data = carryover_data


class Request:
    def __init__(self, month, employee_type, raw_data, override_twoday, is_maxcalls, carryover_data):
        self.month = month
        self.emp_type = employee_type
        self.raw_data = raw_data.fillna('')

        self.roster_list = self.raw_data['Name']
        self.exc_list = self.raw_data.loc[self.raw_data['Exc']=='Yes','Name']
        self.inc_list = self.raw_data.loc[self.roster_list.isin(self.exc_list)]
        self.num_employees = len(self.roster_list)
        self.num_employees_exc = len(self.exc_list)

        if (employee_type == 'Registra'):
            self.ac_list = self.raw_data.loc[self.raw_data['Title']=='AC', 'Name']

        blocked_raw = np.zeros(shape=(self.num_employees, self.month.num_days))
        requested_raw = np.zeros(shape=(self.num_employees, self.month.num_days))
    
        for emp_x in range(0, self.num_employees):  
            for day_x in range(0, self.month.num_days):
                col = str(day_x+1)
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
        max_excluded_raw = self.raw_data.loc[self.raw_data['exc']=='Yes', 'Max']
        max_excluded_data = pd.DataFrame(data = max_excluded_raw.values, columns=['Max_Calls_Exc'], index=self.exc_list)
        self.max_data = max_excluded_data

        # Merge with archive carry over data
        # Extract carry-over values for all employees on current roster
        carryover_data[carryover_data['Name'].isin(self.roster_list)]
        self.carryover_data = carryover_data['Carry_Over']

    # TODO: Add methods to check for the feasibility of the request for:
    # (i) Daily feasibility: i.e. the day has at least one person
    # (ii) For each range of 3 days there is enough people to cover all days
    # (iii) More than one employee requests for a particular day
    # (iv) Not enough blank days for the at least 1 special day requirement to be fulfilled
    # Else returns false with the exact dates that require amendment for model to work

