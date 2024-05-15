import pandas as pd

df_bb = pd.read_csv("Data/mlb_salaries.csv")

# To string allows us to see all the fields
print(df_bb.head().to_string())

df_tor = df_bb[df_bb["teamid"] == "TOR"]
print(f"All players of the Toronto blue jays:\n {df_tor.head(10).to_string()}")

df_agg = df_tor.aggregate("max")
print(df_agg)



max_sal = df_tor["salary"].aggregate("max")

max_play = df_tor[df_tor["salary"] == max_sal]

print(f"\nMax Salary:\n${max_sal}")
print(max_play)
print("\n PLayer %s has max salary $%.2f" % (max_play["player_name"].values[0], max_sal))


