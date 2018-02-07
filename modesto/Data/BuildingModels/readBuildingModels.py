import pandas as pd
import numpy as np


def get_file_names():
    filenames = ['Gebouweigenschappen SFH_D_1_2zone.xlsx',
                 'Gebouweigenschappen SFH_D_2_2zone.xlsx',
                 'Gebouweigenschappen SFH_D_3_2zone.xlsx',
                 'Gebouweigenschappen SFH_D_4_2zone.xlsx',
                 'Gebouweigenschappen SFH_D_5_2zone.xlsx',
                 'Gebouweigenschappen SFH_D_5_ins 2zone.xlsx',
                 'Gebouweigenschappen SFH_SD_1_2zone.xlsx',
                 'Gebouweigenschappen SFH_SD_2_2zone.xlsx',
                 'Gebouweigenschappen SFH_SD_3_2zone.xlsx',
                 'Gebouweigenschappen SFH_SD_4_2zone.xlsx',
                 'Gebouweigenschappen SFH_SD_5 2zone.xlsx',
                 'Gebouweigenschappen SFH_SD_5_Ins 2zone.xlsx',
                 'Gebouweigenschappen SFH_T_1_2zone.xlsx',
                 'Gebouweigenschappen SFH_T_2_2zone.xlsx',
                 'Gebouweigenschappen SFH_T_3_2zone.xlsx',
                 'Gebouweigenschappen SFH_T_4_2zone.xlsx',
                 'Gebouweigenschappen SFH_T_5 2zone.xlsx',
                 'Gebouweigenschappen SFH_T_5_ins 2zone.xlsx']

    return filenames


def get_sheet_names():

    sheets = ['Gebouwgegevens Tabula 2zone', 'Tabula RefULG 1', 'Tabula RefULG 2']

    return sheets


def read_excel_sheet(filename, sheet):

    if filename not in get_file_names():
        raise ValueError('{} is not a valid filename'.format(filename))
    if sheet not in get_sheet_names():
        raise ValueError('{} is not a valid sheet name'.format(sheet))
    if '5' in filename and (sheet == 'Tabula RefULG 1' or sheet == 'Tabula RefULG 2'):
        print 'Warning: No Building models are available for {}.{}'.format(filename, sheet)
        return None

    df = pd.read_excel(io=filename, sheet_name=sheet, skiprows=2)

    columns_to_be_removed = range(0, list(df).index('SolNESW THEO')) + \
                            range(list(df).index('SolNESW THEO')+4, len(df.columns))
    df = df.drop(df.columns[columns_to_be_removed], axis=1)

    df.columns = ['type', 'param', '=', 'value']

    df = df[np.isfinite(df['value'])]
    df = df.set_index('param', drop=True)

    return df

index = read_excel_sheet('Gebouweigenschappen SFH_D_1_2zone.xlsx', 'Gebouwgegevens Tabula 2zone').index

for file_name in get_file_names():
    for sheet_name in get_sheet_names():
        print '\nExcel file: ', file_name
        print 'SHeet name: ', sheet_name
        df = read_excel_sheet(file_name, sheet_name)
        if df is None:
            pass
        elif not index.equals(df.index):
            raise Exception('{} in {} does not give a correct result'.format(sheet_name, file_name))

print 'Test succeeded'