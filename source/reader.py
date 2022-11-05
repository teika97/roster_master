import os
import re
import datetime as dt
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


#%% Read data for model input

#Gets file date from the file
#Args : 
    #file_name (str): Name of the file
    #file_regex (str): Regex expression to specify file name pattern to extract file_date
    #date_regex (str): Regex expression to specify date pattern in file_name
    #return_type (str): Return type of file_date as string or date-time object
#Returns :
    #date as date-time object or string as extracted from file_name
def get_date(file_name, file_regex, date_regex, return_type = "dt"):
    
    file_date = re.search(file_regex, file_name).group(1)
    
    if (return_type == "str"):
        return file_date
    else:
        return dt.datetime.strptime(file_date, date_regex)


#Gets files with the latest date in specified folder
#Args:
    #folder_path_name (str): Directory path of folder storing data
    #file_regex (str): Regex expression to specify file name pattern to extract file_date
    #date_regex (str): Regex expression to specify date pattern in file_name
#Returns:
    #date string and file object corresponding to the file with the latest date
def get_latest(folder_name, file_regex, date_regex):

    folder_path_name = os.path.join(os.getcwd(), folder_name)
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

#Read all required file from file path into dataframe
#Args:
    #cwd (str): Current working directory
    #setting (str): Model settings
    #employee_type (str): 'Reg' - Only Reg, 'Con' - Only Consultant
    #input_type (str): 'Y' - First tab of gsheet, '1' - CSV file with latest month, '2' - CSV file with selected month, '3' - Selected tab of gsheet
    #selected_value (str): Value of selected month or tab for options 2/3 for input_type
#Returns:
    #raw input data in file as a dataframe and roster period
    #if employee_type = 'Reg' then returns the request and roster archive data for only registra
    #if employee_type = 'Con' then returns the request and roster archive data for only consultant
def read_data(cwd, setting, employee_type, input_type, selected_value=None):

    # For checking
    if not ((employee_type == "Reg") | (employee_type == "Con")):
        raise Exception("Input error: Invalid value for employee_type")

    # input_type is only for request data, roster archive data is always extracted through a single excel file from the default folder "data"
    roster_archive_path = os.path.join(cwd, setting.req_folder_name, setting.roster_archive_file_name)

    # Authorizing the program to pull from gsheet
    try:
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(cwd, setting.req_folder_name, setting.passcode_json_file_name), scope)
        client = gspread.authorize(creds)
    except:
        raise Exception("Authorization error: Unable to set up authorization to pull from gsheet")

    if (employee_type == 'Reg'):

        #Read roster archive data for registra with fixed format i.e. multi-index
        roster_archive_r = pd.read_excel(roster_archive_path, sheet_name='Reg', header=[0,1], index_col=[0])

        # First tab of gsheet
        if (input_type == '1'):
            sheet_r = client.open('Call requests (Reg)')
            sheet_instance_r = sheet_r.get_worksheet(0)
            sheet_name_r = sheet_instance_r.title
            if (dt.datetime.strptime(sheet_name_r, "%b %Y")):
                print(sheet_name_r + " from gsheet downloaded")
                file_date_str = sheet_name_r
            else:
                raise Exception("Incorrect input format: Tab name in first registra gsheet must be in Mon YYYY format")
            input_r = pd.DataFrame.from_dict(sheet_instance_r.get_all_records(head=2)) # Skip first row due to formatting of gsheet
        
        # CSV File with latest month
        elif (input_type == '2'):
            file_date_str, latest_file_r = get_latest(setting.req_folder_name, setting.req_file_regex_r, setting.req_date_regex)
            print("Latest file found: " + file_date_str)     
            input_r = pd.read_csv(latest_file_r, skiprows = 1) # Note skip_rows = 1 is to match offline excel format to online gsheet

        # CSV File with selected month
        elif (input_type == '3'):
            req_file_name_r = setting.req_file_prefix_r + selected_value + ".csv"
            req_file_path_r = os.path.join(cwd, setting.req_folder_name, req_file_name_r)
            if (dt.datetime.strptime(selected_value, "%b %Y")):
                file_date_str = selected_value
            else:
                raise Exception("Incorrect month format inputted: Month format must be in Mon YYYY format")            
            input_r = pd.read_csv(req_file_path_r, skiprows = 1)
            print(file_date_str + " file found.")     

        # Selected tab of gsheet
        elif (input_type == '4'):
            sheet_r = client.open('Call requests (Reg)')
            sheet_instance_r = sheet_r.worksheet(selected_value)
            sheet_name_r = selected_value
            if (dt.datetime.strptime(sheet_name_r, "%b %Y")):
                print(sheet_name_r + " from gsheet downloaded")
                file_date_str = sheet_name_r
            else:
                raise Exception("Incorrect input format: Tab name in first registra gsheet must be in Mon YYYY format")
            input_r = pd.DataFrame.from_dict(sheet_instance_r.get_all_records(head=2)) # Skip first row due to formatting of gsheet

        else:
            raise Exception("Input error: Invalid value for input_type")

    if (employee_type == 'Con'):

        # Read roster archive data for consultant with fixed format i.e. multi-index
        roster_archive_c = pd.read_excel(roster_archive_path, sheet_name='Consultant', header=[0,1], index_col=[0])

        # First tab of gsheet
        if (input_type == '1'):
            sheet_c = client.open('Call requests (Consultant)')
            sheet_instance_c = sheet_c.get_worksheet(0)
            sheet_name_c = sheet_instance_c.title
            if (dt.datetime.strptime(sheet_name_c, "%b %Y")):
                print(sheet_name_c + " from gsheet downloaded")
                file_date_str = sheet_name_c
            else:
                raise Exception("Incorrect input format: Tab name in first consultant gsheet must be in Mon YYYY format")
            input_c = pd.DataFrame.from_dict(sheet_instance_c.get_all_records(head=2)) # Skip first row due to formatting of gsheet
        
        # CSV File with latest month
        elif (input_type == '2'):
            file_date_str, latest_file_c = get_latest(setting.req_folder_name, setting.req_file_regex_c, setting.req_date_regex)
            input_c = pd.read_csv(latest_file_c, skiprows = 1)

        # CSV File with selected month
        elif (input_type == '3'):
            req_file_name_c = setting.req_file_prefix_c + selected_value + ".csv"
            req_file_path_c = os.path.join(cwd, setting.req_folder_name, req_file_name_c)
            if (dt.datetime.strptime(selected_value, "%b %Y")):
                file_date_str = selected_value
            else:
                raise Exception("Incorrect month format inputted: Month format must be in Mon YYYY format")
            input_c = pd.read_csv(req_file_path_c, skiprows = 1) # Note skip_rows = 1 is to match offline excel format to online gsheet
            print(file_date_str + " file found.")     

        # Selected tab of gsheet
        elif (input_type == '4'):
            sheet_c = client.open('Call requests (Consultant)')
            sheet_instance_c = sheet_c.worksheet(selected_value)
            sheet_name_c = selected_value
            if (dt.datetime.strptime(sheet_name_c, "%b %Y")):
                print(sheet_name_c + " from gsheet downloaded")
                file_date_str = sheet_name_c
            else:
                raise Exception("Incorrect input format: Tab name in first registra gsheet must be in Mon YYYY format")
            input_c = pd.DataFrame.from_dict(sheet_instance_c.get_all_records(head=2)) # Skip first row due to formatting of gsheet

        else:
            raise Exception("Input error: Invalid value for input_type")

    if (employee_type == 'Reg'):
        return file_date_str, input_r, roster_archive_r
    elif (employee_type == 'Con'):
        return file_date_str, input_c, roster_archive_c
    else:
        raise Exception("Input error: Invalid value for employee_type")
    

