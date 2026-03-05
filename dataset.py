import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def makeDataset(filePath):
    df = pd.read_csv(filePath, names=['Unsorted'])
    return df

myData = makeDataset("Evo_Initial_BCI_Data/2026-27-01_Evo_Run04_FiveSets_Gain12.csv")
pd.set_option('display.max_columns', None)
myNewData = myData['Unsorted'].str.split('\t', expand=True)
myNewData = myNewData.astype(float)
myNewData = myNewData.rename(columns = {
    0: "Sample Index",
    1: "EXG Channel 0",
    2: "EXG Channel 1",
    3: "EXG Channel 2",
    4: "EXG Channel 3",
    5: "EXG Channel 4",
    6: "EXG Channel 5",
    7: "EXG Channel 6",
    8: "EXG Channel 7",
    9: "Accel Channel 0",
    10: "Accel Channel 1",
    11: "Accel Channel 2",
    12: "Not Used",
    13: "Digital Channel 0 (D11)",
    14: "Digital Channel 1 (D12)",
    15: "Digital Channel 2 (D13)",
    16: "Digital Channel 3 (D17)",
    17: "Not Used",
    18: "Digital Channel 4 (D18)",
    19: "Analog Channel 0",
    20: "Analog Channel 1",
    21: "Analog Channel 2",
    22: "Timestamp",
    23: "Marker Channel",
    24: "Timestamp (Formatted)",
})

lowest = myNewData.iloc[0, 22]
myNewData['Timestamp'] = myNewData['Timestamp'] - lowest
myNewData['Label'] = (round(myNewData['Timestamp'], 0) % 10) >= 5
print(myNewData)

# df = myNewData
df = load_iris(as_frame=True)
x = df.drop(columns=['target'])
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)