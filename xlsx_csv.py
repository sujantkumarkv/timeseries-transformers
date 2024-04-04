import pandas as pd

# ../data/ is needed when running wrt scripts/
xlsx_path = "data/xlsx" 
csv_path = "data/csv"

stations = ["arkulagad", "belur", "chikmangalur", "hanball", "kikkeri", "shantig"]

def xlsx_to_csv(xlsx_path, csv_path):
    for station in stations:
        xlsx_file = xlsx_path + f"/rainfall/{station}.xlsx"
        df = pd.read_excel(xlsx_file)

        # Write DataFrame to a .csv file
        df.to_csv(f"{csv_path}/rainfall/{station}.csv", index=False)

# Usage
xlsx_to_csv(xlsx_path, csv_path)