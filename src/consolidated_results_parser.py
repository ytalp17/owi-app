import pandas
import variables as variables

gender = variables.GENDER
consolidated_file_path = f"app_data/{gender}/consolidated_results_{gender}.csv"
consolidated_data = pandas.read_csv(consolidated_file_path)
columns = list(consolidated_data.columns)

race_dict = {col: [] for col in columns}

for (index, row) in consolidated_data.iterrows():
    if row.athlete_name == "athlete_name":
        pass

    elif row.athlete_name == "end":
        file_name = row.event
        df = pandas.DataFrame(race_dict)
        df.to_csv(f"app_data/{gender}/results/{file_name}.csv", index=False)
        print(f"{file_name}.csv created")
        race_dict = {column: [] for column in columns}

    else:
        for col in columns:
            race_dict[col].append(row[col])





