import numpy as np
import pandas as pd
# Preprocessing

def to_float(x):
    x = x.replace('$', '')
    x = x.replace(',', '')
    x = float(x)
    return x

def zip_5d(Zip, State):
    # fix zip code with error
    zero_head = ['CT','MA','ME','NH','NJ','NY','PR','RI','VT','VI','AE','AE']
    if (Zip == 9999) or (Zip == 99999):
        zip_code = '99999'
    elif len(str(Zip)) == 4:
        if State in zero_head:
            zip_code = '0' + str(Zip)
        else:
            zip_code = '99999'
    elif len(str(Zip)) == 3:
        if State in zero_head:
            zip_code = '00' + str(Zip)
        else:
            zip_code = '99999'
    elif len(str(Zip)) < 3:
        zip_code = '99999'
    else:
        zip_code = str(Zip)
    return zip_code

def clean_LowDoc(x):
    """""
    LowDoc (Y = Yes, N = No): In order to process more loans efficiently, 
    a "LowDoc Loan" program was implemented where loans under $150,000 can be
    processed using a one-page application. "Yes" indicates loans with a one-page 
    application, and "No" indicates loans with more information attached to the 
    application. In this dataset, 87.31% are coded as N (No) and 12.31% as Y (Yes) 
    for a total of 99.62%. It is worth noting that 0.38% have other values 
    (0, 1, A, C, R, S); these are data entry errors.
    """""
    if x == 'Y':
        return 1
    elif x == 'N':
        return 0
    else:
        return np.NaN
    
def fix_naics(x):
    """""
    NAICS (North American Industry Classification System): This is a 2- through 
    6-digit hierarchical classification system used by Federal statistical 
    agencies in classifying business establishments for the collection, analysis, 
    and presentation of statistical data describing the U.S. economy. The first 
    two digits of the NAICS classification represent the economic sector.
    """""
    if x == 0:
        naics = '999999'
    else:
        naics = str(x)
    return naics

def naics_sector(x):
    if x == '999999':
        naics = '999999'
    else:
        naics = int(str(x)[:2])
        if (naics >= 31) and (naics <= 33):
            naics = '31'
        elif (naics >= 44) and (naics <= 45):
            naics = '44'
        elif (naics >= 48) and (naics <= 49):
            naics = '48'
        else:
            naics = str(naics)
    return naics

def clean_RevLineCr(x):
    if x == 'Y':
        return 1
    else:
        return 0

def RealEstate(x):
    if x >= 240:
        return 1
    else:
        return 0

def franchise(x): 
    # FranchiseCode:    Franchise Code 00000 or 00001 = No Franchise
    if (x == 1) or (x == 0):
        return 0
    else:
        return 1    

# def SBA_ratio(SBA_Appv, GrAppv):
#     if type(SBA_Appv) != float:
#         SBA_Appv = to_float(SBA_Appv)
#     if type(GrAppv) != float:
#         GrAppv = to_float(GrAppv)
#     return SBA_Appv/GrAppv    

def naics_defaut_rate(x):
    default_dict = {'11': 9, '21': 8, '22': 14, '23': 23, '31': 19, '32': 16, '33': 14, '42': 19, '44': 22, '45': 23, '48': 27, '49': 23, '51': 25, '52': 28, '53': 29, '54': 19, '55': 10, '56': 24, '61': 24, '62': 10, '71': 21, '72': 22, '81': 20, '92': 15}
    naics = str(x)[:2]
    if naics not in default_dict.keys():
        default_rate = np.NaN
    else:
        default_rate = default_dict[naics]
    return default_rate    
    
def default(x):
    if x == 'CHGOFF':
        return 1
    else:
        return 0

# Create features
def company_suffix(x):
    # classify the business based on the abbreviation after the business name
    l = x.split()[-1].replace('.', '').upper()
    if l in ['C', 'L', 'I', 'A', 'S', 'D', 'M']:
        if 'L L C' in x.upper():
            l = 'LLC'
        elif len(x) == 1:
            l = l
        else:
            l = x[:-1].split()[-1].replace('.', '').upper()
            
    if l in ['INC', 'INCORPORATED']: # incorporated
        abb = 'INC'
    elif l in ['CO', 'COMPANY']:
        if 'LTD' in x.upper():
            abb = 'LTD'
        else:
            abb = 'CO'
    elif l in ['IN']:
        abb = 'IN'
    elif l in ['LLC']: # limited liability companies
        abb = 'LLC'
    elif l in ['LLP']: # limited liability companies
        abb = 'LLP'
    elif l in ['LL']: # limited liability companies
        abb = 'LL'  
    elif l in ['LTD']: # "LTD" or "Ltd." stands for "limited"
        abb = 'LTD'
    elif l in ['PC']: # professional corporation such as medicine, law and accounting
        abb = 'PC'
    elif l in ['PLLC']: # professional limited liability company
        abb = 'PLLC'
    elif l in ['PA']: # professional association
        abb = 'PA'
    elif l in ['CORP', 'CORPORATION']:
        abb = 'CORP'
    elif l in ['ASSOC', 'ASSOCIATES']:
        abb = 'ASSOC'
    elif l in ['DDS']:
        abb = 'DDS'
    elif l in ['MD', 'CLINIC']:
        abb = 'CLINIC'
    elif l in ['HOSPITAL']:
        abb = 'HOSPITAL'      
    elif l in ['SALON']:
        abb = 'SALON' 
    elif l in ['CONSTRUCTION']:
        abb = 'CONSTRUCTION'
    elif l in ['CLEANERS']:
        abb = 'CLEANERS' 
    elif l in ['LAUNDRY', ]:
        abb = 'LAUNDRY'
    elif l in ['DMD']:
        abb = 'DMD' 
    elif l in ['ENTERPRISES']:
        abb = 'ENTERPRISES'
    elif l in ['CLUB', 'FOUNDATION', 'FUND', 'INSTITUTE', 'SOCIETY', 'UNION', 'SYNDICATE']:
        abb = 'Non-Profit'
    elif l in ['RESTAURANT', 'CAFE', 'GRILL']:
        abb = 'RESTAURANT'
    elif l in ['MARKET', 'MART', 'SUPERMARKET', 'GROCERY', 'DELI', 'PIZZA', 'BAR']:
        abb = 'MARKET'
    elif l in ['SERVICES', 'SERVICE', 'SERV', 'SERVIC']:
        abb = 'SERVICE'   
    elif l in ['FARM']:
        abb = 'FARM'
    elif l in ['HOTEL', 'MOTEL', 'INN']:
        abb = 'HOTEL'   
    else:
        abb = 'NO SUFFIX'
    return abb

def loan_age(year, loan_record_dict):
    if pd.isnull(loan_record_dict):
        return np.NaN
    else:
        if year <= min(loan_record_dict.keys()):
            age = 0
        else:
            age = year - min(loan_record_dict.keys())
        return age

# loan_age(2006 ,loan_dict)
def previous_loan(year, loan_record_dict):
    if pd.isnull(loan_record_dict):
        return np.NaN  
    else:
        loan_list = filter(lambda x: x < year, loan_record_dict.keys())
    #     print loan_list
        record = 0
        for i in loan_list:
            record = record + loan_record_dict[i]
        return record
    
def default_times(year, default_record_dict):
    if pd.isnull(default_record_dict):
        return 0
    else:
        default_list = filter(lambda x: x < year, default_record_dict.keys())
        record = 0
        for i in default_list:
            record = record + default_record_dict[i]
        return record

# Job related features

def expanding(x):
    # company is expanding if the loan is used to create jobs
    if x > 0:
        return 1
    else:
        return 0
    
def retaining(x):
    # company is in troubale if the loan is used to retain jobs    
    if x > 0:
        return 1
    else:
        return 0    
    
def expanding_ratio(CreateJob, NoEmp):
    if pd.isnull(CreateJob) or pd.isnull(NoEmp) or (NoEmp == 0):
        return 'No info'
    else:
        r = CreateJob / NoEmp
        if r > 1:
            return 'Double size!'
        elif r > 0.5:
            return 'Over 50%'
        elif r > 0.1:
            return '10%~50%'
        elif r > 0:
            return 'Less than 10%'
        else:
            return 'No change'

def retaining_ratio(RetainedJob, NoEmp):
    if pd.isnull(RetainedJob) or pd.isnull(NoEmp) or (NoEmp == 0):
        return 'No info'
    else:
        r = RetainedJob / NoEmp
        if r > 0.8:
            return 'Over 80%'
        elif r > 0.5:
            return '50~80%'
        elif r > 0.2:
            return '20~50%'
        elif r > 0:
            return 'Less than 20%'
        else:
            return 'No change'
    
    
# MISC

def fix_missing_state(nat):
    missing_state = {49244: 'NY',
                     264664: 'CA',
                     306274: 'CA',
                     328526: 'KS',
                     351072: 'TX',
                     366139: 'FL',
                     366158: 'WI',
                     367007: 'WI',
                     379174: 'UT',
                     385418: 'MO',
                     869948: 'TX',
                     871847: 'TX',
                     885335: 'TX'}
    for key in missing_state.keys():
        nat.loc[key, 'State'] = missing_state[key]
#         print key, nat.loc[key].State
    return nat

# Extract features

def extract_train_features(features, drop, categorical):
    print('-----> Extract train features <------')
    print('dropping unwanted columns')
    features = features.drop(drop, axis=1)

    print('transforming categorical variables')
    dict_categorical = {}
    for col in categorical:
        cat = pd.Categorical(features[col])
        new_col = col[:-1]+'INT'
        if col[-2] == '_':
            new_col = col[:-1]+'INT'
        else:
            new_col = col+'_INT'
        features.loc[:, new_col] = cat.codes
        dict_categorical[col] = dict([(k, v) for v, k in enumerate(cat.categories)])
    features = features.drop(categorical, axis=1)
    print('done')
    return dict_categorical, features

def extract_test_features(test, drop, categorical, dict_categorical):
    print('-----> Extract test features <------')
    print('dropping unwanted columns')
    test=test.drop(drop, axis=1)
    print('transforming categorical variabless')
    for col in categorical:
        new_col = col[:-1]+'INT'
        if col[-2] == '_':
            new_col = col[:-1]+'INT'
        else:
            new_col = col+'_INT'
        test[new_col] = test[col].map(dict_categorical[col])
        test[new_col].fillna(-1, inplace=True)
    test=test.drop(categorical, axis=1)
    print('done')
    return test