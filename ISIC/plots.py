import matplotlib.pyplot as plt
import pandas as pd

truth_csv = pd.read_csv('truth.csv')

print("ISIC diagnosis statistics")
counted = []
for column in truth_csv.columns[1:]:
    counted.append(truth_csv[column].value_counts()[1])
    print(column, ":", counted[-1])
    print()

plt.figure(figsize=(10, 5))
plt.bar(truth_csv.columns[1:], counted)
plt.savefig("ISIC_truths.png")
plt.close()


metadata_csv = pd.read_csv('metadata.csv')
#print(metadata_csv.info())

counted = []
for column in metadata_csv.columns[6:10]:
    counted.append(metadata_csv[column].value_counts())
    #print(counted[-1])
    print(counted[-1].values)
    print(counted[-1].index)
    print()
    plt.figure(figsize=(10, 5))
    plt.bar(counted[-1].index, counted[-1].values)
    plt.savefig(f"ISIC_metadata_{column}.png")
    plt.close()