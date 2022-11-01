import pandas as pd
pd.options.display.width = None

data = pd.read_csv('datasets/space_can_be_a_dangerous_place.csv')
print(data[['black_hole_is_near', 'buggers_were_noticed', 'nearby_system_has_planemo', 'dangerous']].corr())