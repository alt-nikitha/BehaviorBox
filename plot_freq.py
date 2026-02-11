import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/home/nsrikant/.cache/n_moreearly_olmo3_7b_unigram/olmo_validation_texts/unigram_freqs.csv')

# Plotting the histogram of the 'frequency' column
plt.figure(figsize=(10, 6))
plt.hist(df['count'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Token Frequencies')
plt.xlabel('Frequency')
plt.ylabel('Number of Tokens')
plt.yscale('log') # Often necessary for corpus data
plt.grid(axis='y', alpha=0.75)
plt.savefig('frequency_histogram.png')