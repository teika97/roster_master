import os
import re
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import calendar
import holidays
import datetime as dt
from datetime import datetime, timedelta
from storage import Month, Archive, Request

req_file_regex_c = "Call requests (Consultant) - Nov 2022.csv"
req_file_regex_r = "Call requests (Reg) - Nov 2022.csv"
roster_archive_file_name = "Roster_Archive.xlsx"
roster_period = 'Nov 2022'

cwd = os.getcwd()
input_c = pd.read_csv(os.path.join(cwd, 'data', req_file_regex_c), skiprows = 1)
input_r = pd.read_csv(os.path.join(cwd, 'data', req_file_regex_r), skiprows = 1)

roster_archive_c = pd.read_excel(os.path.join(cwd, 'data', roster_archive_file_name), sheet_name='Consultant', header=[0,1], index_col=[0])
roster_archive_r = pd.read_excel(os.path.join(cwd, 'data', roster_archive_file_name), sheet_name='Reg', header=[0,1], index_col=[0])

# Public holidays in Singapore
sg_holidays = []
for date in holidays.Singapore(years=datetime.strptime(roster_period, '%b %Y').year).items():
    sg_holidays.append(date[0])

month_data = Month(roster_period, sg_holidays)
archive_c = Archive('Consultant', roster_archive_c)
archive_r = Archive('Registra', roster_archive_r)
request_c = Request(month_data, 'Consultant', input_c, False, False, archive_c.carryover_data)
request_r = Request(month_data, 'Registra', input_c, False, False, archive_r.carryover_data)
