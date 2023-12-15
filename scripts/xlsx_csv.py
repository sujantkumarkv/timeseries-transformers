import pandas as pd

# ../data/ is needed when running wrt scripts/
xlsx_path = "data/xlsx" 
csv_path = "data/csv"

def xlsx_to_csv(xlsx_path, csv_path):
    # Load spreadsheet
    xlsx_file = xlsx_path + "/streamflow.xlsx"
    df = pd.read_excel(xlsx_file)

    # Write DataFrame to a .csv file
    df.to_csv(f"{csv_path}/streamflow.csv", index=False)

# Usage
xlsx_to_csv(xlsx_path, csv_path)