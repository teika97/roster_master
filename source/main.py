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
import gspread
from oauth2client.service_account import ServiceAccountCredentials


cwd = os.getcwd()

roster_archive_file_name = "Roster_Archive.xlsx"
roster_period = 'Nov 2022'

roster_archive_c = pd.read_excel(os.path.join(cwd, 'data', roster_archive_file_name), sheet_name='Consultant', header=[0,1], index_col=[0])
roster_archive_r = pd.read_excel(os.path.join(cwd, 'data', roster_archive_file_name), sheet_name='Reg', header=[0,1], index_col=[0])
# Public holidays in Singapore
sg_holidays = []
for date in holidays.Singapore(years=datetime.strptime(roster_period, '%b %Y').year).items():
    sg_holidays.append(date[0])

roster_month = Month(roster_period, sg_holidays)

input_type = input("Import from gsheet or folder: ")
if (input_type == "folder"):
    req_file_regex_c = "Call requests (Consultant) - Nov 2022.csv"
    req_file_regex_r = "Call requests (Reg) - Nov 2022.csv"
    input_c = pd.read_csv(os.path.join(cwd, 'data', req_file_regex_c), skiprows = 1)
    input_r = pd.read_csv(os.path.join(cwd, 'data', req_file_regex_r), skiprows = 1)
elif (input_type == "gsheet"):
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(cwd,'source','passcode_key.json'), scope)
    client = gspread.authorize(creds)

    sheet_c = client.open('Call requests (Consultant)')
    sheet_r = client.open('Call requests (Reg)')

    sheet_instance_c = sheet_c.get_worksheet(0)
    sheet_instance_r = sheet_r.get_worksheet(0)

    sheet_name_c = sheet_instance_c.title
    sheet_name_r = sheet_instance_r.title

    input_c = pd.DataFrame.from_dict(sheet_instance_c.get_all_records(head=2))
    input_r = pd.DataFrame.from_dict(sheet_instance_r.get_all_records(head=2))

archive_c = Archive('Consultant', roster_archive_c)
archive_r = Archive('Registra', roster_archive_r)
request_c = Request(roster_month, 'Consultant', input_c, False, False, archive_c.carryover_data)
request_r = Request(roster_month, 'Registra', input_r, False, False, archive_r.carryover_data)

request_c.print_details()
request_r.print_details()