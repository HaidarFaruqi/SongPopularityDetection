# 🎓 Tutorial Lengkap: Cara Membuat & Menggunakan Setiap Fitur

## 📋 Daftar Isi
1. [Persiapan Data](#persiapan-data)
2. [Artist Features - Target Encoding](#artist-features)
3. [Audio Features - Engineering](#audio-features)
4. [Temporal Features - Time-based](#temporal-features)
5. [Track Name Features - Text Processing](#track-name-features)
6. [NLP Features - Lyrics Processing](#nlp-features)
7. [Interaction Features - Feature Combinations](#interaction-features)
8. [Categorical Encoding](#categorical-encoding)
9. [Feature Preparation untuk Model](#feature-preparation)

---

## 1️⃣ Persiapan Data

### **STEP 0: Load dan Inspect Data**

```python
import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Lihat kolom yang tersedia
print("\nKolom di dataset:")
print(train_df.columns.tolist())

# Output:
# ['track_id', 'track_name', 'artists', 'release_year', 'track_genre',
#  'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
#  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
#  'duration_ms', 'explicit', 'lyrics', 'popularity']
```

**Lihat beberapa baris:**
```python
train_df.head(3)

# Output (contoh):
#   track_id  track_name           artists  release_year  track_genre  ...  popularity
#   0         Blinding Lights      The Weeknd    2020      synth-pop    ...  89
#   1         Shape of You         Ed Sheeran    2017      pop          ...  87
#   2         Someone Like You     Adele         2011      pop          ...  82
```

**Inspect data types:**
```python
train_df.dtypes

# Output:
# track_id              object
# track_name            object
# artists               object
# release_year          int64
# track_genre           object
# danceability          float64
# energy                float64
# ...
# popularity            int64
```

**Check missing values:**
```python
missing = train_df.isnull().sum()
print(missing[missing > 0])

# Output (contoh):
# lyrics    1523
# (artinya ada 1,523 lagu tanpa lyrics)
```

**Basic statistics:**
```python
train_df['popularity'].describe()

# Output:
# count    89740.000
# mean        45.234
# std         24.567
# min          0.000
# 25%         28.000
# 50%         46.000
# 75%         64.000
# max        100.000
```

---

## 2️⃣ Artist Features - Target Encoding

### **❓ MENGAPA Fitur Ini Penting?**

**Analogi Sederhana:**
```
Bayangkan kamu lihat 2 lagu baru:
Lagu A: dari Taylor Swift
Lagu B: dari artist tidak dikenal

SEBELUM DENGAR, mana yang lebih likely populer?
→ Lagu A! Karena Taylor Swift = brand power!

Ini yang di-capture oleh artist_avg_pop
```

**Data-Driven Evidence:**
```python
# Lihat perbedaan artist popularity
artist_stats = train_df.groupby('artists')['popularity'].agg(['mean', 'count'])
artist_stats = artist_stats.sort_values('mean', ascending=False)

print(artist_stats.head(10))

# Output (contoh):
#                    mean   count
# artists
# Bad Bunny         88.3     142
# Taylor Swift      85.7     238
# The Weeknd        83.2     156
# Dua Lipa          82.1     127
# Harry Styles      81.5     98
# ...

print(artist_stats.tail(10))

# Output (contoh):
#                         mean   count
# Random Indie Artist 1   12.3     3
# Unknown Band            11.8     2
# ...

# RANGE: 88.3 - 11.8 = 76.5 points difference!
# Ini HUGE impact!
```

---

### **🔧 CARA MEMBUAT: Target Encoding (OUT-OF-FOLD)**

**SALAH 1: Simple Mean (LEAKAGE!)**
```python
# ❌ JANGAN SEPERTI INI! (Data leakage)
artist_avg = train_df.groupby('artists')['popularity'].mean()
train_df['artist_avg_pop'] = train_df['artists'].map(artist_avg)

# Kenapa salah?
# Model "melihat" popularity dari row yang sama!
# Ini cheating dan akan overfit!
```

**BENAR 1: Global Mean (Safe tapi kurang optimal)**
```python
# ✅ Aman tapi tidak maksimal
artist_avg = train_df.groupby('artists')['popularity'].mean()
global_mean = train_df['popularity'].mean()

# Untuk train: gunakan artist average
train_df['artist_avg_pop'] = train_df['artists'].map(artist_avg)

# Untuk test: gunakan yang sama + fallback ke global mean
test_df['artist_avg_pop'] = test_df['artists'].map(artist_avg).fillna(global_mean)

# Masalah: train masih bisa "lihat" nilai sendiri
```

**BENAR 2: OUT-OF-FOLD Encoding (BEST!)**
```python
from sklearn.model_selection import KFold

# Persiapan
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
train_df['artist_avg_pop'] = 0.0  # Initialize
global_mean = train_df['popularity'].mean()

# OUT-OF-FOLD encoding
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
    # Hitung average dari TRAIN fold saja
    train_fold = train_df.iloc[train_idx]
    artist_avg = train_fold.groupby('artists')['popularity'].mean()

    # Apply ke VAL fold
    val_fold_artists = train_df.iloc[val_idx]['artists']
    train_df.loc[val_idx, 'artist_avg_pop'] = val_fold_artists.map(artist_avg).fillna(global_mean)

# Untuk test: gunakan average dari SELURUH train
artist_avg_full = train_df.groupby('artists')['popularity'].mean()
test_df['artist_avg_pop'] = test_df['artists'].map(artist_avg_full).fillna(global_mean)
```

**Penjelasan OUT-OF-FOLD:**
```
Dataset split jadi 5 fold:

FOLD 1 (20%):  val
FOLD 2 (20%):  train
FOLD 3 (20%):  train  ← Hitung average dari FOLD 2,3,4,5
FOLD 4 (20%):  train  ← Apply ke FOLD 1 (yang tidak dipakai untuk hitung)
FOLD 5 (20%):  train

Iterasi:
- Fold 1 = val → avg dari fold 2,3,4,5
- Fold 2 = val → avg dari fold 1,3,4,5
- dst...

Hasilnya: TIDAK ADA LEAKAGE!
Setiap row menggunakan average dari row LAIN
```

**Simplified Version (untuk praktis):**
```python
# Cara simple yang cukup bagus (tanpa OOF):
# Hitung average popularity per artist
artist_avg_map = train_df.groupby('artists')['popularity'].mean()
artist_count_map = train_df.groupby('artists').size()

# Global mean sebagai fallback
global_mean_pop = train_df['popularity'].mean()
global_mean_count = train_df['artists'].value_counts().mean()

# Apply ke train
train_df['artist_avg_pop'] = train_df['artists'].map(artist_avg_map).fillna(global_mean_pop)
train_df['artist_song_count'] = train_df['artists'].map(artist_count_map).fillna(global_mean_count)

# Apply ke test (MUST use train statistics!)
test_df['artist_avg_pop'] = test_df['artists'].map(artist_avg_map).fillna(global_mean_pop)
test_df['artist_song_count'] = test_df['artists'].map(artist_count_map).fillna(global_mean_count)

print(f"Artist avg pop range: {train_df['artist_avg_pop'].min():.1f} - {train_df['artist_avg_pop'].max():.1f}")
# Output: Artist avg pop range: 12.3 - 88.3
```

**Verify hasil:**
```python
# Check beberapa artist
sample_artists = ['Taylor Swift', 'The Weeknd', 'Unknown Artist (tidak ada di train)']

for artist in sample_artists:
    if artist in artist_avg_map.index:
        print(f"{artist}: avg_pop = {artist_avg_map[artist]:.1f}")
    else:
        print(f"{artist}: avg_pop = {global_mean_pop:.1f} (fallback)")

# Output:
# Taylor Swift: avg_pop = 85.7
# The Weeknd: avg_pop = 83.2
# Unknown Artist: avg_pop = 45.2 (fallback)
```

---

### **📊 MENGAPA Artist Features Bekerja?**

**Visualisasi Impact:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot: artist_avg_pop vs popularity
plt.figure(figsize=(10, 6))
sample = train_df.sample(5000)
plt.scatter(sample['artist_avg_pop'], sample['popularity'], alpha=0.3, s=10)
plt.plot([0, 100], [0, 100], 'r--', label='Perfect correlation')
plt.xlabel('Artist Avg Popularity')
plt.ylabel('Song Popularity')
plt.title('Artist Brand Effect')
plt.legend()
plt.show()

# Correlation
corr = train_df['artist_avg_pop'].corr(train_df['popularity'])
print(f"Correlation: {corr:.3f}")
# Output: Correlation: 0.652

# Ini STRONGEST single feature!
```

**Contoh Real:**
```python
# Taylor Swift case study
taylor_songs = train_df[train_df['artists'] == 'Taylor Swift']

print(f"Taylor Swift total songs: {len(taylor_songs)}")
print(f"Avg popularity: {taylor_songs['popularity'].mean():.1f}")
print(f"Min popularity: {taylor_songs['popularity'].min()}")
print(f"Max popularity: {taylor_songs['popularity'].max()}")

# Output:
# Taylor Swift total songs: 238
# Avg popularity: 85.7
# Min popularity: 72
# Max popularity: 96

# Bahkan lagu "terburuk" Taylor Swift (72) masih tinggi!
# Ini brand power!
```

---

## 3️⃣ Audio Features - Engineering

### **A. Energy × Danceability Interaction**

**❓ MENGAPA Fitur Ini?**

**Konsep:**
```
Lagu dengan energy TINGGI + danceability TINGGI = CLUB HITS!

Contoh:
- "Uptown Funk" (Bruno Mars): energy 0.9, dance 0.8 → viral!
- "Blinding Lights" (The Weeknd): energy 0.8, dance 0.8 → viral!

Lagu dengan HANYA satu tinggi:
- Classical music: energy 0.8, dance 0.2 → niche
- Slow ballad: energy 0.2, dance 0.3 → niche

INTERACTION captures this non-linear effect!
```

**🔧 CARA MEMBUAT:**

```python
# Simple multiplication
train_df['energy_x_dance'] = train_df['energy'] * train_df['danceability']
test_df['energy_x_dance'] = test_df['energy'] * test_df['danceability']

# Verify distribution
print(train_df['energy_x_dance'].describe())

# Output:
# count    89740.000
# mean         0.432
# std          0.215
# min          0.000
# 25%          0.256
# 50%          0.441
# 75%          0.612
# max          1.000
```

**Visualisasi:**
```python
# Heatmap: energy vs danceability → popularity
pivot = train_df.copy()
pivot['energy_bin'] = pd.cut(pivot['energy'], bins=5, labels=['VLow', 'Low', 'Med', 'High', 'VHigh'])
pivot['dance_bin'] = pd.cut(pivot['danceability'], bins=5, labels=['VLow', 'Low', 'Med', 'High', 'VHigh'])

heatmap_data = pivot.groupby(['energy_bin', 'dance_bin'])['popularity'].mean().unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn')
plt.title('Avg Popularity: Energy × Danceability')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.show()

# Akan terlihat: kotak kanan atas (High energy × High dance) = warna hijau (populer!)
```

---

### **B. Duration in Minutes**

**❓ MENGAPA Convert ke Minutes?**

```
duration_ms = 245,678 ms  ← Susah interpret!
duration_min = 4.09 min   ← Lebih mudah dipahami

Model juga lebih mudah belajar:
- Scale lebih kecil (0-10 vs 0-600000)
- Lebih interpretable
```

**🔧 CARA MEMBUAT:**

```python
# Convert milliseconds to minutes
train_df['duration_min'] = train_df['duration_ms'] / 60000
test_df['duration_min'] = test_df['duration_ms'] / 60000

# Check distribution
print(train_df['duration_min'].describe())

# Output:
# count    89740.000
# mean         3.542
# std          1.123
# min          0.333  (20 seconds)
# 25%          2.850
# 50%          3.450
# 75%          4.120
# max         30.250  (outlier: very long song)
```

**Pattern Analysis:**
```python
# Bin by duration
train_df['duration_cat'] = pd.cut(train_df['duration_min'],
                                   bins=[0, 2, 3, 4, 5, 100],
                                   labels=['Very Short', 'Short', 'Normal', 'Long', 'Very Long'])

# Avg popularity by duration
duration_pop = train_df.groupby('duration_cat')['popularity'].agg(['mean', 'count'])
print(duration_pop)

# Output:
#              mean   count
# Very Short   42.3    2,150
# Short        51.2   18,450
# Normal       48.7   45,230  ← SWEET SPOT (3-4 min)
# Long         44.1   20,100
# Very Long    35.8    3,810

# Insight: 3-4 menit optimal!
```

---

### **C. Key + Mode Combination**

**❓ MENGAPA Combine Key & Mode?**

**Konsep Musik:**
```
Key (tangga nada):
- C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- Range: 0-11

Mode:
- 0 = Minor (sedih, dark)
- 1 = Major (happy, bright)

KOMBINASI matters:
- C Major (C + Major) = happy vibe
- C Minor (C + Minor) = sad vibe
- Meskipun sama-sama C, MOOD berbeda!
```

**🔧 CARA MEMBUAT:**

```python
# Combine key and mode
train_df['key_mode'] = train_df['key'].astype(str) + '_' + train_df['mode'].astype(str)
test_df['key_mode'] = test_df['key'].astype(str) + '_' + test_df['mode'].astype(str)

# Check unique combinations
print(f"Unique key_mode combinations: {train_df['key_mode'].nunique()}")
# Output: 24 combinations (12 keys × 2 modes)

# Sample values
print(train_df['key_mode'].value_counts().head(10))

# Output:
# 0_1    9,842  (C Major)
# 2_1    8,765  (D Major)
# 9_1    8,234  (A Major)
# 7_0    7,891  (G Minor)
# ...
```

**Pattern Analysis:**
```python
# Avg popularity by key_mode
key_mode_pop = train_df.groupby('key_mode')['popularity'].mean().sort_values(ascending=False)

print(key_mode_pop.head(10))

# Output (contoh):
# 0_1    47.8  (C Major)
# 2_1    47.2  (D Major)
# 9_0    46.5  (A Minor)
# ...

# Note: Perbedaan kecil, tapi tetap membantu model
```

---

### **D. Tempo Categories (Binning)**

**❓ MENGAPA Binning Tempo?**

**Konsep:**
```
Tempo raw: 0 - 250 BPM (continuous)

Tapi manusia categorize musik:
- Slow: < 90 BPM (ballad, lullaby)
- Moderate: 90-120 BPM (pop, rock)
- Fast: 120-150 BPM (dance, edm)
- Very Fast: > 150 BPM (hardstyle, speed metal)

Binning helps model learn categorical patterns
```

**🔧 CARA MEMBUAT:**

```python
# Create tempo bins
train_df['tempo_category'] = pd.cut(train_df['tempo'],
                                     bins=[0, 90, 120, 150, 250],
                                     labels=['slow', 'moderate', 'fast', 'very_fast'])

test_df['tempo_category'] = pd.cut(test_df['tempo'],
                                    bins=[0, 90, 120, 150, 250],
                                    labels=['slow', 'moderate', 'fast', 'very_fast'])

# Distribution
print(train_df['tempo_category'].value_counts())

# Output:
# moderate     45,231  (50.4%)
# fast         28,456  (31.7%)
# slow         12,345  (13.8%)
# very_fast     3,708   (4.1%)
```

**Pattern Analysis:**
```python
# Avg popularity by tempo category
tempo_pop = train_df.groupby('tempo_category')['popularity'].agg(['mean', 'count'])
print(tempo_pop)

# Output:
#             mean   count
# slow        43.2   12,345
# moderate    46.8   45,231  ← SWEET SPOT
# fast        47.1   28,456
# very_fast   42.5    3,708

# Insight: Moderate-fast optimal (90-150 BPM)
```

---

## 4️⃣ Temporal Features - Time-based

### **A. Years Since Release**

**❓ MENGAPA Fitur Ini Penting?**

**Recency Bias:**
```
Spotify data heavily biased towards RECENT music!

Alasan:
1. Platform baru (founded 2008)
2. More data for recent songs
3. Algorithm promotes new releases
4. Streaming culture = current music
```

**🔧 CARA MEMBUAT:**

```python
# Current year reference
CURRENT_YEAR = 2025

# Calculate years since release
train_df['years_since_release'] = CURRENT_YEAR - train_df['release_year']
test_df['years_since_release'] = CURRENT_YEAR - test_df['release_year']

# Check distribution
print(train_df['years_since_release'].describe())

# Output:
# count    89740.000
# mean        18.234
# std         15.678
# min          0.000  (2025 release)
# 25%          6.000  (2019)
# 50%         13.000  (2012)
# 75%         25.000  (2000)
# max         75.000  (1950)
```

**Pattern Analysis:**
```python
# Bin by recency
train_df['recency_bin'] = pd.cut(train_df['years_since_release'],
                                  bins=[0, 2, 5, 10, 20, 100],
                                  labels=['Very Recent (0-2y)', 'Recent (2-5y)',
                                          'Moderate (5-10y)', 'Old (10-20y)',
                                          'Very Old (20+y)'])

recency_pop = train_df.groupby('recency_bin')['popularity'].agg(['mean', 'count'])
print(recency_pop)

# Output:
#                      mean   count
# Very Recent (0-2y)   56.8   18,200
# Recent (2-5y)        52.3   22,100
# Moderate (5-10y)     48.1   20,500
# Old (10-20y)         42.7   16,300
# Very Old (20+y)      35.4   12,640

# CLEAR TREND: Newer = More Popular!
```

**Visualisasi:**
```python
# Scatter plot
plt.figure(figsize=(12, 6))
sample = train_df.sample(5000)
plt.scatter(sample['years_since_release'], sample['popularity'], alpha=0.3, s=10)
plt.xlabel('Years Since Release')
plt.ylabel('Popularity')
plt.title('Recency Bias in Popularity')

# Trend line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(train_df['years_since_release'],
                                                          train_df['popularity'])
x_line = np.array([0, 75])
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Trend (R²={r_value**2:.3f})')
plt.legend()
plt.show()

# Akan terlihat: downward trend (semakin lama, semakin tidak populer)
```

---

### **B. Decade**

**❓ MENGAPA Decade?**

**Era-Specific Patterns:**
```
Setiap dekade punya karakteristik musik berbeda:

1950s-1960s: Rock n Roll, Motown
1970s: Disco, Funk
1980s: Synth-pop, Hair Metal
1990s: Grunge, Hip-hop rise
2000s: Pop-punk, R&B
2010s: EDM, Trap
2020s: TikTok viral, Hyperpop

Model bisa belajar era patterns!
```

**🔧 CARA MEMBUAT:**

```python
# Extract decade
train_df['decade'] = (train_df['release_year'] // 10) * 10
test_df['decade'] = (test_df['release_year'] // 10) * 10

# Distribution
print(train_df['decade'].value_counts().sort_index())

# Output:
# 1950      850
# 1960    2,100
# 1970    3,200
# 1980    5,800
# 1990    9,500
# 2000   18,200
# 2010   28,500
# 2020   21,600

# Exponential growth!
```

**Pattern Analysis:**
```python
# Avg popularity by decade
decade_pop = train_df.groupby('decade')['popularity'].agg(['mean', 'std', 'count'])
print(decade_pop)

# Output:
#        mean   std   count
# 1950   28.3  18.2     850
# 1960   32.1  19.5   2,100
# 1970   35.8  20.1   3,200
# 1980   39.2  21.5   5,800
# 1990   42.5  22.3   9,500
# 2000   48.7  24.1  18,200
# 2010   52.3  25.8  28,500
# 2020   55.1  26.5  21,600

# LINEAR INCREASE dari decade ke decade!
```

**Visualisasi:**
```python
# Bar chart
plt.figure(figsize=(12, 6))
decade_pop.reset_index().plot(x='decade', y='mean', kind='bar',
                               color='skyblue', edgecolor='black', legend=False)
plt.xlabel('Decade')
plt.ylabel('Avg Popularity')
plt.title('Popularity Evolution by Decade')
plt.xticks(rotation=0)

# Add value labels
for i, v in enumerate(decade_pop['mean']):
    plt.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

plt.show()
```

---

### **C. Binary Flags: is_classic & is_recent_hit**

**❓ MENGAPA Binary Flags?**

**Pattern Recognition:**
```
Model bisa belajar:
- Lagu klasik (< 2000) punya behavior berbeda
- Lagu baru (>= 2020) punya boost dari platform

Binary flag = explicit signal untuk pattern ini
```

**🔧 CARA MEMBUAT:**

```python
# Classic songs (before 2000)
train_df['is_classic'] = (train_df['release_year'] < 2000).astype(int)
test_df['is_classic'] = (test_df['release_year'] < 2000).astype(int)

# Recent hits (2020 onwards)
train_df['is_recent_hit'] = (train_df['release_year'] >= 2020).astype(int)
test_df['is_recent_hit'] = (test_df['release_year'] >= 2020).astype(int)

# Distribution
print(f"Classic songs: {train_df['is_classic'].sum()} ({train_df['is_classic'].mean()*100:.1f}%)")
print(f"Recent hits: {train_df['is_recent_hit'].sum()} ({train_df['is_recent_hit'].mean()*100:.1f}%)")

# Output:
# Classic songs: 21,450 (23.9%)
# Recent hits: 21,600 (24.1%)
```

**Pattern Analysis:**
```python
# Compare popularity
classic_pop = train_df[train_df['is_classic'] == 1]['popularity'].mean()
modern_pop = train_df[train_df['is_classic'] == 0]['popularity'].mean()

recent_pop = train_df[train_df['is_recent_hit'] == 1]['popularity'].mean()
not_recent_pop = train_df[train_df['is_recent_hit'] == 0]['popularity'].mean()

print(f"Classic (pre-2000):    {classic_pop:.1f}")
print(f"Modern (2000+):        {modern_pop:.1f}")
print(f"Difference:            {modern_pop - classic_pop:.1f}")

print(f"\nRecent (2020+):        {recent_pop:.1f}")
print(f"Not recent (pre-2020): {not_recent_pop:.1f}")
print(f"Difference:            {recent_pop - not_recent_pop:.1f}")

# Output:
# Classic (pre-2000):    37.8
# Modern (2000+):        49.2
# Difference:            11.4
#
# Recent (2020+):        55.1
# Not recent (pre-2020): 42.3
# Difference:            12.8

# SIGNIFICANT DIFFERENCES!
```

---

## 5️⃣ Track Name Features - Text Processing

### **A. Track Name Cleaning**

**❓ MENGAPA Perlu Cleaning?**

**Masalah dengan Raw Track Names:**
```
Raw examples:
"Bohemian Rhapsody - Remastered 2011"
"Shape of You (feat. Someone)"
"Blinding Lights - Radio Edit"
"Don't Start Now [Official Audio]"

Noise yang harus dihapus:
- (feat. ...)
- - Remastered ...
- [Official Audio]
- (Radio Edit)

Clean version lebih akurat untuk length/word count
```

**🔧 CARA CLEANING:**

```python
import re

def clean_track_name(name):
    """
    Clean track name dari noise
    """
    if pd.isna(name):
        return ""

    # Convert to lowercase
    clean = str(name).lower()

    # Remove content in parentheses/brackets
    clean = re.sub(r'[\(\[].*?[\)\]]', '', clean)

    # Remove " - feat." and everything after
    clean = clean.split(' - feat.')[0]
    clean = clean.split(' - with')[0]
    clean = clean.split(' - sped up')[0]
    clean = clean.split(' - slowed')[0]
    clean = clean.split(' - remastered')[0]
    clean = clean.split(' - from')[0]
    clean = clean.split(' - radio edit')[0]

    # Strip whitespace
    clean = clean.strip()

    return clean

# Apply cleaning
train_df['track_name_clean'] = train_df['track_name'].apply(clean_track_name)
test_df['track_name_clean'] = test_df['track_name'].apply(clean_track_name)

# Example
print("Before → After:")
for i in range(5):
    print(f"{train_df.iloc[i]['track_name'][:50]:50s} → {train_df.iloc[i]['track_name_clean']}")

# Output:
# Bohemian Rhapsody - Remastered 2011               → bohemian rhapsody
# Shape of You (feat. Someone)                      → shape of you
# Blinding Lights [Official Audio]                  → blinding lights
```

---

### **B. Track Name Length**

**❓ MENGAPA Length Matters?**

**Hypothesis:**
```
Shorter names = catchier, more memorable

Examples:
✅ "Dynamite" (8 chars)
✅ "Butter" (6 chars)
✅ "Levitating" (10 chars)

vs

❌ "Bohemian Rhapsody - Remastered 2011 Version" (43 chars)
❌ "Don't Start Now - Extended Dance Remix" (39 chars)
```

**🔧 CARA MEMBUAT:**

```python
# Calculate length
train_df['track_name_length'] = train_df['track_name_clean'].str.len()
test_df['track_name_length'] = test_df['track_name_clean'].str.len()

# Distribution
print(train_df['track_name_length'].describe())

# Output:
# count    89740.000
# mean        15.234
# std          8.567
# min          1.000  (single letter)
# 25%          9.000
# 50%         14.000
# 75%         20.000
# max         95.000  (very long)
```

**Pattern Analysis:**
```python
# Bin by length
train_df['length_bin'] = pd.cut(train_df['track_name_length'],
                                 bins=[0, 10, 20, 30, 100],
                                 labels=['Very Short (1-10)', 'Short (10-20)',
                                        'Long (20-30)', 'Very Long (30+)'])

length_pop = train_df.groupby('length_bin')['popularity'].agg(['mean', 'count'])
print(length_pop)

# Output:
#                    mean   count
# Very Short (1-10)  54.2   23,450  ← HIGHEST!
# Short (10-20)      48.3   42,100
# Long (20-30)       43.1   18,900
# Very Long (30+)    38.5    5,290

# CLEAR PATTERN: Shorter = Better!
```

**Correlation:**
```python
corr = train_df['track_name_length'].corr(train_df['popularity'])
print(f"Correlation: {corr:.3f}")
# Output: Correlation: -0.118

# Negative = semakin panjang, semakin tidak populer
```

---

### **C. Track Name Word Count**

**🔧 CARA MEMBUAT:**

```python
# Count words (split by space)
train_df['track_name_word_count'] = train_df['track_name_clean'].str.count(' ') + 1
test_df['track_name_word_count'] = test_df['track_name_clean'].str.count(' ') + 1

# Handle empty names
train_df.loc[train_df['track_name_clean'] == '', 'track_name_word_count'] = 0
test_df.loc[test_df['track_name_clean'] == '', 'track_name_word_count'] = 0

# Distribution
print(train_df['track_name_word_count'].value_counts().head(10))

# Output:
# 1    12,340  (single word: "Dynamite", "Butter")
# 2    28,450  (two words: "Blinding Lights", "Shape Of")
# 3    25,100  (three words: "Don't Start Now")
# 4    15,200
# 5     6,780
# ...
```

**Pattern:**
```python
# Avg popularity by word count
word_pop = train_df.groupby('track_name_word_count')['popularity'].mean()

print(word_pop.head(10))

# Output:
# 1    53.7  ← Single word optimal!
# 2    50.1
# 3    47.8
# 4    45.2
# 5    43.1
# 6    41.5
# ...

# DECREASING trend
```

---

## 6️⃣ NLP Features - Lyrics Processing

### **❓ MENGAPA NLP untuk Lyrics?**

**Konsep:**
```
Lyrics = TEXT data

Masalah:
- Model tidak bisa baca text langsung
- Perlu convert ke NUMBERS

Solutions:
1. TF-IDF: Convert text → numeric vectors
2. SVD: Reduce dimensions (500 → 20)
```

### **A. TF-IDF Vectorization**

**Penjelasan TF-IDF:**

```
TF-IDF = Term Frequency × Inverse Document Frequency

Contoh sederhana:

LAGU 1: "love love love baby baby"
LAGU 2: "heartbreak tears cry sadness"
LAGU 3: "party dance club night"

STEP 1: Term Frequency (TF)
- "love" di Lagu 1: 3/5 = 0.6
- "love" di Lagu 2: 0/4 = 0.0
- "love" di Lagu 3: 0/4 = 0.0

STEP 2: Inverse Document Frequency (IDF)
- "love" muncul di 1/3 lagu
- IDF = log(3/1) = 0.477

STEP 3: TF × IDF
- "love" di Lagu 1: 0.6 × 0.477 = 0.286
- "the" (muncul semua): IDF rendah → score rendah
- "heartbreak" (unique): IDF tinggi → score tinggi
```

**🔧 CARA IMPLEMENTASI:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Persiapan: Fill missing lyrics
train_df['lyrics'] = train_df['lyrics'].fillna('')
test_df['lyrics'] = test_df['lyrics'].fillna('')

# Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=500,        # Ambil top 500 kata
    min_df=5,                # Kata harus muncul min di 5 lagu
    max_df=0.8,              # Buang kata yang muncul di >80% lagu
    ngram_range=(1, 2),      # Unigram (1 kata) dan Bigram (2 kata)
    stop_words='english'     # Buang stopwords ("the", "is", "a", dll)
)

# Fit on train, transform both
train_tfidf = tfidf.fit_transform(train_df['lyrics'])
test_tfidf = tfidf.transform(test_df['lyrics'])

print(f"TF-IDF matrix shape (train): {train_tfidf.shape}")
# Output: (89740, 500)

print(f"TF-IDF matrix shape (test): {test_tfidf.shape}")
# Output: (22435, 500)
```

**Inspect TF-IDF vocabulary:**
```python
# Top words by importance
feature_names = tfidf.get_feature_names_out()
print(f"\nTop 20 words/bigrams:")
print(feature_names[:20])

# Output (example):
# ['love', 'baby', 'night', 'oh', 'know', 'like', 'come', 'want',
#  'heart', 'time', 'got', 'never', 'yeah', 'need', 'feel', 'girl',
#  'say', 'away', 'see', 'make']

# Bigrams (2-word phrases)
bigrams = [f for f in feature_names if ' ' in f]
print(f"\nSample bigrams:")
print(bigrams[:10])

# Output (example):
# ['love you', 'oh oh', 'baby baby', 'wanna be', 'feel like',
#  'let go', 'come on', 'got to', 'right now', 'know that']
```

---

### **B. TruncatedSVD (Dimensionality Reduction)**

**❓ MENGAPA SVD?**

**Problem:**
```
TF-IDF = 500 features

Issues:
1. Too many features → overfitting risk
2. SPARSE (banyak nilai 0)
3. Redundant (banyak kata similar meaning)

Example sparse matrix:
Song 1: [0.5, 0, 0, 0.3, 0, 0, 0, 0.8, 0, ...]
Song 2: [0, 0, 0.2, 0, 0.4, 0, 0, 0, 0, ...]
        ↑ Mostly zeros!

SVD Solution:
500 features → 20 dense features
```

**Konsep SVD:**
```
SVD = Singular Value Decomposition

Analogi:
Imagine 500-dimensional space → project to 20-dimensional space
while keeping MOST information

Like PCA, but for sparse matrices

Result:
- Topic 0: Love songs dimension
- Topic 1: Party songs dimension
- Topic 2: Sad songs dimension
- ...
- Topic 19: Other patterns dimension
```

**🔧 CARA IMPLEMENTASI:**

```python
from sklearn.decomposition import TruncatedSVD

# Initialize SVD
svd = TruncatedSVD(
    n_components=20,      # Reduce to 20 dimensions
    random_state=42
)

# Fit and transform
train_lyrics_svd = svd.fit_transform(train_tfidf)
test_lyrics_svd = svd.transform(test_tfidf)

print(f"SVD output shape (train): {train_lyrics_svd.shape}")
# Output: (89740, 20)

print(f"SVD output shape (test): {test_lyrics_svd.shape}")
# Output: (22435, 20)

# Check explained variance
explained_var = svd.explained_variance_ratio_.sum()
print(f"\nExplained variance: {explained_var:.2%}")
# Output: Explained variance: 34.56%

# Artinya: 20 komponen capture 34.56% informasi dari 500 features
# Ini cukup bagus untuk compressed representation!
```

**Inspect SVD components:**
```python
# Each component captures a "topic"
for i in range(5):
    # Get top words for this component
    component = svd.components_[i]
    top_indices = component.argsort()[-10:][::-1]  # Top 10 words
    top_words = [feature_names[idx] for idx in top_indices]

    print(f"\nTopic {i}:")
    print(", ".join(top_words))

# Output (example):
# Topic 0: love, baby, heart, feel, need, want, know, girl, night, like
#         → LOVE SONGS TOPIC
#
# Topic 1: party, dance, club, night, DJ, beat, floor, move, music, vibe
#         → PARTY/CLUB TOPIC
#
# Topic 2: cry, tears, sad, hurt, pain, lonely, goodbye, lost, break, heart
#         → SAD SONGS TOPIC
#
# Topic 3: hey, yeah, oh, come, let, wanna, gonna, get, right, now
#         → UPBEAT/CALL-TO-ACTION TOPIC
#
# Topic 4: god, pray, faith, believe, soul, spirit, heaven, lord, grace
#         → RELIGIOUS/SPIRITUAL TOPIC
```

---

### **C. Add to DataFrame**

```python
# Create column names
lyrics_cols = [f'lyrics_feature_{i}' for i in range(20)]

# Convert to DataFrame
train_lyrics_df = pd.DataFrame(train_lyrics_svd,
                                columns=lyrics_cols,
                                index=train_df.index)

test_lyrics_df = pd.DataFrame(test_lyrics_svd,
                               columns=lyrics_cols,
                               index=test_df.index)

# Concatenate with original dataframes
train_df = pd.concat([train_df, train_lyrics_df], axis=1)
test_df = pd.concat([test_df, test_lyrics_df], axis=1)

print(f"\nNew columns added:")
print(train_df.columns[-20:].tolist())

# Output:
# ['lyrics_feature_0', 'lyrics_feature_1', ..., 'lyrics_feature_19']
```

**Verify lyrics features:**
```python
# Check distribution of first lyrics feature
print(train_df['lyrics_feature_0'].describe())

# Output:
# count    89740.000
# mean         0.000  (centered)
# std          0.512
# min         -2.345
# 25%         -0.234
# 50%          0.012
# 75%          0.298
# max          3.567

# Check correlation with popularity
corr = train_df['lyrics_feature_0'].corr(train_df['popularity'])
print(f"Correlation with popularity: {corr:.3f}")
# Output: Correlation with popularity: 0.142

# Positive correlation = this topic relates to popular songs!
```

---

## 7️⃣ Interaction Features - Feature Combinations

### **❓ MENGAPA Interaction Features?**

**Non-Linear Effects:**
```
Linear thinking:
popularity = w1 × artist_avg_pop + w2 × danceability

But reality:
Taylor Swift + danceable song ≠ just sum!
Taylor Swift + danceable song = MEGA VIRAL!

Non-linear effect = MULTIPLICATION
```

**Concrete Example:**
```
SCENARIO A:
artist_avg_pop = 85 (Taylor Swift)
danceability = 0.8
Linear: 85 + 0.8 = 85.8
Interaction: 85 × 0.8 = 68.0 ← NEW SIGNAL!

SCENARIO B:
artist_avg_pop = 42 (Unknown)
danceability = 0.8
Linear: 42 + 0.8 = 42.8
Interaction: 42 × 0.8 = 33.6 ← Much lower!

Model learns: Popular artist × good audio = amplified effect!
```

---

### **A. Artist × Danceability**

**🔧 CARA MEMBUAT:**

```python
# Simple multiplication
train_df['artist_x_dance'] = train_df['artist_avg_pop'] * train_df['danceability']
test_df['artist_x_dance'] = test_df['artist_avg_pop'] * test_df['danceability']

# Distribution
print(train_df['artist_x_dance'].describe())

# Output:
# count    89740.000
# mean        28.456
# std         18.234
# min          0.000
# 25%         14.560
# 50%         26.780
# 75%         40.120
# max         88.300  (max artist × max dance = 100 × 0.883)
```

**Pattern Analysis:**
```python
# High interaction songs
high_interaction = train_df.nlargest(10, 'artist_x_dance')
print(high_interaction[['track_name', 'artists', 'artist_avg_pop',
                        'danceability', 'artist_x_dance', 'popularity']])

# Output (example):
#              track_name        artists  artist_avg  dance  interact  popularity
# Levitating   Dua Lipa             82.1    0.85      69.8       87
# Don't Start Now Dua Lipa          82.1    0.79      64.9       85
# Dynamite     BTS                  85.2    0.75      63.9       89
# ...

# Pattern: High interaction = high popularity!
```

**Visualisasi:**
```python
# 3D scatter (if possible) or 2D heatmap
plt.figure(figsize=(10, 6))
sample = train_df.sample(5000)

scatter = plt.scatter(sample['artist_x_dance'],
                     sample['popularity'],
                     c=sample['danceability'],
                     cmap='viridis',
                     alpha=0.5, s=20)

plt.xlabel('Artist × Danceability Interaction')
plt.ylabel('Popularity')
plt.title('Interaction Effect on Popularity')
plt.colorbar(scatter, label='Danceability')

# Trend line
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(sample['artist_x_dance'],
                                        sample['popularity'])
x_line = np.array([0, 90])
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r--', linewidth=2,
         label=f'Trend (R²={r**2:.3f})')
plt.legend()
plt.show()
```

---

### **B. Artist × Energy**

**Similar logic:**

```python
train_df['artist_x_energy'] = train_df['artist_avg_pop'] * train_df['energy']
test_df['artist_x_energy'] = test_df['artist_avg_pop'] * test_df['energy']
```

---

## 8️⃣ Categorical Encoding

### **❓ MENGAPA Perlu Encoding?**

**Problem:**
```
Model ML hanya bisa baca NUMBERS, tidak bisa baca TEXT

track_genre = "pop"      ← Cannot use!
track_genre = 45         ← Can use!
```

**🔧 CARA: Label Encoding**

```python
from sklearn.preprocessing import LabelEncoder

# List categorical features
categorical_features = ['track_genre', 'key_mode', 'tempo_category', 'decade']
categorical_features_raw = [f for f in categorical_features if f in train_df.columns]

# Store encoders
label_encoders = {}
encoded_cat_features = []

for col in categorical_features_raw:
    print(f"\nEncoding {col}...")

    # Initialize encoder
    le = LabelEncoder()

    # IMPORTANT: Fit on COMBINED train + test
    # Kenapa? Agar semua kategori ter-encode
    combined_series = pd.concat([
        train_df[col].astype(str),
        test_df[col].astype(str)
    ])

    le.fit(combined_series)

    # Transform
    train_df[col + '_encoded'] = le.transform(train_df[col].astype(str))
    test_df[col + '_encoded'] = le.transform(test_df[col].astype(str))

    # Store encoder
    label_encoders[col] = le
    encoded_cat_features.append(col + '_encoded')

    # Show mapping
    print(f"  Unique values: {len(le.classes_)}")
    print(f"  Example mapping: {dict(list(zip(le.classes_[:5], range(5))))}")

# Output:
# Encoding track_genre...
#   Unique values: 114
#   Example mapping: {'acoustic': 0, 'afrobeat': 1, 'alt-rock': 2, 'alternative': 3, 'ambient': 4}
#
# Encoding key_mode...
#   Unique values: 24
#   Example mapping: {'0_0': 0, '0_1': 1, '1_0': 2, '1_1': 3, '2_0': 4}
# ...
```

**Verify encoding:**
```python
# Check some examples
print("\nExample encoding results:")
for i in range(5):
    print(f"track_genre: {train_df.iloc[i]['track_genre']:15s} → {train_df.iloc[i]['track_genre_encoded']}")

# Output:
# track_genre: pop             → 45
# track_genre: rock            → 78
# track_genre: hip-hop         → 32
# track_genre: electronic      → 25
# track_genre: jazz            → 38
```

---

## 9️⃣ Feature Preparation untuk Model

### **A. Feature Selection**

```python
# Get all numeric features
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude kolom yang TIDAK digunakan untuk modeling
exclude_cols = [
    'popularity',        # Target (jangan masuk features!)
    'track_id',          # ID (tidak informatif)
    'track_name',        # Text (sudah di-extract jadi length/word_count)
    'artists',           # Text (sudah di-encode jadi artist_avg_pop)
    'lyrics',            # Text (sudah di-extract jadi lyrics_feature_*)
    'release_year'       # Raw year (sudah jadi decade, years_since_release)
]

numeric_features = [f for f in numeric_features if f not in exclude_cols]

print(f"Total numeric features: {len(numeric_features)}")
# Output: Total numeric features: 45
```

---

### **B. Remove Duplicate Features**

**CRITICAL: Avoid duplicate features!**

```python
# PROBLEM: Setelah encoding, *_encoded sudah jadi numeric
# Jadi bisa masuk ke numeric_features DAN encoded_cat_features
# Ini DUPLICATE → Error!

# SOLUTION: Remove *_encoded from numeric_features
numeric_features_clean = [f for f in numeric_features
                         if not f.endswith('_encoded')]

# Combine
features = numeric_features_clean + encoded_cat_features

print(f"\nFinal features: {len(features)}")
# Output: Final features: 49

# Verify no duplicates
assert len(features) == len(set(features)), "DUPLICATE FEATURES FOUND!"
print("✓ No duplicates")
```

---

### **C. Imputation (Handle Missing Values)**

```python
from sklearn.impute import SimpleImputer

# Initialize imputer (median strategy)
imputer = SimpleImputer(strategy='median')

# Fit on train, transform both
train_df[features] = imputer.fit_transform(train_df[features])
test_df[features] = imputer.transform(test_df[features])

print("\n✓ Missing values imputed")

# Verify no missing
print(f"Missing in train: {train_df[features].isnull().sum().sum()}")
print(f"Missing in test: {test_df[features].isnull().sum().sum()}")
# Output:
# Missing in train: 0
# Missing in test: 0
```

---

### **D. Final Feature Matrix**

```python
# Prepare X and y
X_train = train_df[features]
y_train = train_df['popularity']

X_test = test_df[features]

print("\n✅ FEATURE PREPARATION COMPLETE!")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Output:
# Train shape: (89740, 49)
# Test shape: (22435, 49)

# List all features
print("\n📋 Final 49 features:")
for i, feat in enumerate(features, 1):
    print(f"{i:2d}. {feat}")

# Output:
#  1. danceability
#  2. energy
#  3. loudness
#  4. speechiness
#  ...
# 49. tempo_category_encoded
```

---

## 🎯 Summary: Complete Pipeline

```python
# STEP-BY-STEP RECAP:

# 1. Artist Features (Target Encoding)
artist_avg_map = train_df.groupby('artists')['popularity'].mean()
train_df['artist_avg_pop'] = train_df['artists'].map(artist_avg_map).fillna(global_mean)
test_df['artist_avg_pop'] = test_df['artists'].map(artist_avg_map).fillna(global_mean)

# 2. Audio Features (Engineering)
train_df['energy_x_dance'] = train_df['energy'] * train_df['danceability']
train_df['duration_min'] = train_df['duration_ms'] / 60000
train_df['key_mode'] = train_df['key'].astype(str) + '_' + train_df['mode'].astype(str)

# 3. Temporal Features
train_df['years_since_release'] = 2025 - train_df['release_year']
train_df['decade'] = (train_df['release_year'] // 10) * 10
train_df['is_recent_hit'] = (train_df['release_year'] >= 2020).astype(int)

# 4. Track Name Features
clean_name = train_df['track_name'].str.lower().str.replace(r'[\(\[].*?[\)\]]', '', regex=True)
train_df['track_name_length'] = clean_name.str.len()
train_df['track_name_word_count'] = clean_name.str.count(' ') + 1

# 5. NLP Features (Lyrics)
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words='english')
train_tfidf = tfidf.fit_transform(train_df['lyrics'])
svd = TruncatedSVD(n_components=20)
train_lyrics = svd.fit_transform(train_tfidf)
# Add to df...

# 6. Interaction Features
train_df['artist_x_dance'] = train_df['artist_avg_pop'] * train_df['danceability']

# 7. Categorical Encoding
le = LabelEncoder()
train_df['track_genre_encoded'] = le.fit_transform(train_df['track_genre'])

# 8. Feature Selection & Preparation
features = numeric_features_clean + encoded_cat_features
X = train_df[features]
y = train_df['popularity']

# 9. Model Training
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
model.fit(X, y)

# DONE! RMSE ~16.05
```

---

## 📚 Key Takeaways

1. **Artist features = KING** (28.5% importance)
2. **Target encoding needs OUT-OF-FOLD** to prevent leakage
3. **Interaction features** capture non-linear effects
4. **NLP (TF-IDF + SVD)** extracts meaning from lyrics
5. **Temporal features** capture recency bias
6. **Feature engineering > complex models**
7. **Clean data > fancy algorithms**

---

**Made with ❤️ by Tim AhThatsHot**
**RMSE 16.05 Achievement Unlocked! 🏆**
