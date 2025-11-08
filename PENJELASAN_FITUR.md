# 📚 PENJELASAN FITUR-FITUR SISTEM PREDIKSI POPULARITAS LAGU

## 🎯 Daftar Isi
1. [Fitur Artist (Target Encoding)](#1-fitur-artist-target-encoding)
2. [Fitur Audio](#2-fitur-audio)
3. [Fitur Temporal](#3-fitur-temporal)
4. [Fitur Track Name](#4-fitur-track-name)
5. [Fitur Interaction](#5-fitur-interaction)
6. [Fitur Lyrics (NLP)](#6-fitur-lyrics-nlp)
7. [Fitur Kategorik](#7-fitur-kategorik)

---

## 1. Fitur Artist (Target Encoding)

### 🎤 `artist_avg_pop`
**Definisi**: Rata-rata popularitas lagu dari seorang artist

**Cara Kerja**:
```python
# Hitung mean popularitas per artist dari training data
artist_popularity_map = train_df.groupby('artists')['popularity'].mean()

# Apply ke semua data
train_df['artist_avg_pop'] = train_df['artists'].map(artist_popularity_map)
```

**Contoh**:
- Taylor Swift memiliki rata-rata popularitas 75
- Lagu baru dari Taylor Swift akan mendapat fitur `artist_avg_pop = 75`
- Artist yang tidak dikenal akan mendapat nilai global mean (rata-rata semua lagu)

**Mengapa Penting?**
- Artist terkenal cenderung menghasilkan lagu populer
- Fitur ini menangkap "reputasi" atau "brand value" dari artist
- Biasanya menjadi fitur paling penting dalam model

**Catatan**:
- Menggunakan **target encoding** (leak-free karena dihitung dari training data saja)
- Untuk artist baru/tidak dikenal, gunakan global mean sebagai fallback

---

### 📊 `artist_song_count`
**Definisi**: Jumlah total lagu dari seorang artist dalam dataset

**Cara Kerja**:
```python
# Hitung jumlah lagu per artist
artist_song_count_map = train_df.groupby('artists').size()

# Apply ke semua data
train_df['artist_song_count'] = train_df['artists'].map(artist_song_count_map)
```

**Contoh**:
- Artist A memiliki 50 lagu dalam dataset → `artist_song_count = 50`
- Artist B memiliki 2 lagu dalam dataset → `artist_song_count = 2`

**Mengapa Penting?**
- Artist produktif mungkin memiliki pola popularitas berbeda
- Menangkap apakah artist adalah "mainstream" (banyak lagu) atau "indie" (sedikit lagu)
- Korelasi dengan exposure dan fan base

---

## 2. Fitur Audio

### ⚡ `energy_x_dance`
**Definisi**: Interaksi antara energy dan danceability

**Cara Kerja**:
```python
df['energy_x_dance'] = df['energy'] * df['danceability']
```

**Interpretasi**:
- **Tinggi**: Lagu energetic dan danceable → cocok untuk party, gym
- **Rendah**: Lagu calm dan less danceable → cocok untuk relaksasi

**Contoh**:
- Dance Pop: energy=0.8, danceability=0.9 → `energy_x_dance = 0.72` (TINGGI)
- Classical: energy=0.3, danceability=0.2 → `energy_x_dance = 0.06` (RENDAH)

**Mengapa Penting?**
- Lagu dengan kombinasi high energy + high danceability sering populer (EDM, Pop)
- Non-linear interaction lebih informatif daripada fitur individual

---

### ⏱️ `duration_min`
**Definisi**: Durasi lagu dalam menit (konversi dari milliseconds)

**Cara Kerja**:
```python
df['duration_min'] = df['duration_ms'] / 60000
```

**Interpretasi**:
- **Pendek** (< 3 min): Modern pop, radio-friendly
- **Sedang** (3-5 min): Standar kebanyakan lagu
- **Panjang** (> 5 min): Progressive rock, classical, atau jam sessions

**Mengapa Penting?**
- Lagu pendek lebih radio-friendly → potensi popularitas lebih tinggi
- Tren modern: lagu makin pendek (streaming era)

---

### 🎹 `key_mode`
**Definisi**: Kombinasi key (nada dasar) dan mode (major/minor)

**Cara Kerja**:
```python
df['key_mode'] = df['key'].astype(str) + '_' + df['mode'].astype(str)
```

**Contoh**:
- Key=0 (C), Mode=1 (Major) → `key_mode = '0_1'` (C Major)
- Key=5 (F), Mode=0 (Minor) → `key_mode = '5_0'` (F Minor)

**Mengapa Penting?**
- Kombinasi key dan mode mempengaruhi mood lagu
- Major cenderung "happy", Minor cenderung "sad"
- Beberapa key lebih populer di genre tertentu

---

### 🏃 `tempo_category`
**Definisi**: Kategori tempo lagu (slow/moderate/fast/very_fast)

**Cara Kerja**:
```python
df['tempo_category'] = pd.cut(df['tempo'],
                              bins=[0, 90, 120, 150, 250],
                              labels=['slow', 'moderate', 'fast', 'very_fast'])
```

**Kategori**:
- **Slow** (0-90 BPM): Ballads, lo-fi
- **Moderate** (90-120 BPM): Pop, R&B
- **Fast** (120-150 BPM): Dance, Rock
- **Very Fast** (150+ BPM): EDM, Hardstyle

**Mengapa Penting?**
- Tempo mempengaruhi mood dan use case lagu
- Genre tertentu memiliki tempo range spesifik
- Binning membuat model lebih robust

---

## 3. Fitur Temporal

### 📅 `years_since_release`
**Definisi**: Umur lagu (dalam tahun) sejak rilis hingga sekarang (2025)

**Cara Kerja**:
```python
df['years_since_release'] = 2025 - df['release_year']
```

**Contoh**:
- Lagu rilis 2023 → `years_since_release = 2` (lagu baru)
- Lagu rilis 1990 → `years_since_release = 35` (lagu klasik)

**Mengapa Penting?**
- Lagu baru cenderung lebih populer (recency bias)
- Lagu klasik yang bertahan punya popularitas stabil
- Menangkap temporal trend

---

### 📆 `decade`
**Definisi**: Dekade rilis lagu (1990, 2000, 2010, 2020)

**Cara Kerja**:
```python
df['decade'] = (df['release_year'] // 10) * 10
```

**Contoh**:
- 1997 → `decade = 1990`
- 2015 → `decade = 2010`

**Mengapa Penting?**
- Setiap dekade punya karakteristik musik berbeda
- Trend popularitas berbeda per era
- Genre dominan berubah tiap dekade

---

### 🏛️ `is_classic`
**Definisi**: Flag untuk lagu klasik (rilis sebelum tahun 2000)

**Cara Kerja**:
```python
df['is_classic'] = (df['release_year'] < 2000).astype(int)
```

**Nilai**:
- `1`: Lagu klasik (pre-2000)
- `0`: Lagu modern (2000+)

**Mengapa Penting?**
- Lagu klasik yang bertahan hingga sekarang biasanya "timeless"
- Pola popularitas lagu klasik berbeda dengan lagu modern

---

### 🔥 `is_recent_hit`
**Definisi**: Flag untuk lagu baru (rilis >= 2020)

**Cara Kerja**:
```python
df['is_recent_hit'] = (df['release_year'] >= 2020).astype(int)
```

**Nilai**:
- `1`: Lagu sangat baru (2020+)
- `0`: Lagu lama

**Mengapa Penting?**
- Lagu baru punya momentum dari marketing dan promotion
- Streaming era (2020+) punya karakteristik berbeda

---

## 4. Fitur Track Name

### 📝 `track_name_length`
**Definisi**: Panjang nama lagu (jumlah karakter) setelah cleaning

**Cara Kerja**:
```python
# Clean track name
clean_name = df['track_name'].str.lower()
clean_name = clean_name.str.replace(r'[\(\[].*?[\)\]]', '', regex=True)
clean_name = clean_name.str.split(' - feat.').str[0]
# ... cleaning lainnya

df['track_name_length'] = clean_name.str.len()
```

**Contoh**:
- "Shape of You" → length = 12
- "Bohemian Rhapsody" → length = 17

**Mengapa Penting?**
- Nama lagu pendek lebih mudah diingat
- Trend modern: nama lagu makin pendek dan catchy

---

### 🔢 `track_name_word_count`
**Definisi**: Jumlah kata dalam nama lagu

**Cara Kerja**:
```python
df['track_name_word_count'] = clean_name.str.count(' ') + 1
```

**Contoh**:
- "Hello" → word count = 1
- "Stairway to Heaven" → word count = 3

**Mengapa Penting?**
- Satu kata lebih catchy dan memorable
- Korelasi dengan simplicity dan marketability

---

## 5. Fitur Interaction

### 🎨 `artist_x_dance`
**Definisi**: Interaksi antara popularitas artist dan danceability lagu

**Cara Kerja**:
```python
df['artist_x_dance'] = df['artist_avg_pop'] * df['danceability']
```

**Interpretasi**:
- Artist populer + lagu danceable = potensi hit tinggi
- Artist tidak populer + lagu danceable = potensi viral

**Mengapa Penting?**
- Menangkap non-linear relationship
- Artist populer yang bikin lagu danceable cenderung sukses besar
- Lebih informatif daripada fitur individual

---

### ⚡ `artist_x_energy`
**Definisi**: Interaksi antara popularitas artist dan energy lagu

**Cara Kerja**:
```python
df['artist_x_energy'] = df['artist_avg_pop'] * df['energy']
```

**Interpretasi**:
- Artist populer + lagu energetic = formula sukses
- Menangkap "signature sound" dari artist

**Mengapa Penting?**
- Artist terkenal dengan energetic songs sering dominan chart
- Interaction menangkap synergy effect

---

## 6. Fitur Lyrics (NLP)

### 📖 `lyrics_feature_0` hingga `lyrics_feature_19`
**Definisi**: Fitur dari pemrosesan lyrics menggunakan NLP (20 komponen)

**Cara Kerja**:

1. **TF-IDF Vectorization**:
```python
tfidf = TfidfVectorizer(
    max_features=500,      # Ambil 500 kata terpenting
    min_df=5,              # Kata minimal muncul di 5 lagu
    max_df=0.8,            # Buang kata yang muncul di >80% lagu
    ngram_range=(1, 2),    # Unigram dan bigram
    stop_words='english'   # Buang stopwords
)
```

2. **Dimensionality Reduction (SVD)**:
```python
svd = TruncatedSVD(n_components=20)
lyrics_features = svd.fit_transform(tfidf_matrix)
```

**Apa itu TF-IDF?**
- **TF** (Term Frequency): Seberapa sering kata muncul dalam lyrics
- **IDF** (Inverse Document Frequency): Seberapa unik kata tersebut
- Formula: `TF-IDF = TF × IDF`
- Kata umum (seperti "the", "is") mendapat score rendah
- Kata spesifik dan rare mendapat score tinggi

**Apa itu SVD?**
- Truncated Singular Value Decomposition
- Teknik dimensionality reduction (kompres fitur)
- Mengubah 500 fitur TF-IDF menjadi 20 komponen utama
- Menangkap pola semantik dan topik dalam lyrics

**Contoh Interpretasi**:
- `lyrics_feature_0`: Mungkin menangkap "tema cinta"
- `lyrics_feature_1`: Mungkin menangkap "tema party"
- `lyrics_feature_2`: Mungkin menangkap "tema sad/melancholic"

**Mengapa Penting?**
- Lyrics mempengaruhi emotional connection dengan pendengar
- Tema tertentu lebih populer (cinta, heartbreak, party)
- Kompleksitas lyrics berkorelasi dengan genre dan audience

**Explained Variance**:
- 20 komponen biasanya menangkap 30-50% variance
- Trade-off antara information retention dan dimensionality

---

## 7. Fitur Kategorik

### 🎵 `track_genre_encoded`
**Definisi**: Genre lagu yang di-encode menjadi angka

**Cara Kerja**:
```python
le = LabelEncoder()
df['track_genre_encoded'] = le.fit_transform(df['track_genre'])
```

**Contoh**:
- "pop" → 0
- "rock" → 1
- "hip-hop" → 2

**Mengapa Penting?**
- Genre adalah salah satu predictor terkuat popularitas
- Pop, Hip-Hop, EDM cenderung lebih populer
- Metal, Classical cenderung niche

---

### 🎹 `key_mode_encoded`
**Definisi**: Kombinasi key dan mode yang di-encode

**Mengapa Penting?**
- Setiap kombinasi key-mode punya karakteristik unik
- LightGBM bisa capture pattern dalam categorical features

---

### 🏃 `tempo_category_encoded`
**Definisi**: Kategori tempo yang di-encode

**Mengapa Penting?**
- Lebih robust daripada raw tempo value
- Binning mengurangi noise

---

### 📅 `decade_encoded`
**Definisi**: Dekade yang di-encode

**Mengapa Penting?**
- Menangkap era musik
- Setiap dekade punya trend popularitas berbeda

---

## 📊 Ringkasan Kategori Fitur

| Kategori | Jumlah Fitur | Contoh | Importance Level |
|----------|--------------|--------|------------------|
| **Artist** | 2 | artist_avg_pop, artist_song_count | ⭐⭐⭐⭐⭐ (Sangat Tinggi) |
| **Audio** | 15+ | energy, danceability, loudness, tempo, dll | ⭐⭐⭐⭐ (Tinggi) |
| **Temporal** | 4 | years_since_release, decade, is_classic, is_recent_hit | ⭐⭐⭐ (Sedang) |
| **Lyrics** | 20 | lyrics_feature_0 hingga lyrics_feature_19 | ⭐⭐⭐ (Sedang-Tinggi) |
| **Track Name** | 2 | track_name_length, track_name_word_count | ⭐⭐ (Rendah-Sedang) |
| **Interaction** | 2 | artist_x_dance, artist_x_energy | ⭐⭐⭐⭐ (Tinggi) |
| **Genre** | 1 | track_genre_encoded | ⭐⭐⭐⭐ (Tinggi) |

---

## 🎯 Feature Importance Insights

Berdasarkan hasil training, biasanya urutan importance:

1. **`artist_avg_pop`** - Popularitas artist adalah predictor terkuat
2. **`track_genre_encoded`** - Genre sangat mempengaruhi popularitas
3. **`years_since_release`** - Temporal factor penting
4. **`energy_x_dance`** - Interaction features sangat informatif
5. **Audio features** (energy, danceability, loudness, dll)
6. **Lyrics features** - Menangkap semantic content
7. **Track name features** - Contribution lebih kecil tapi tetap berguna

---

## 💡 Tips Feature Engineering

1. **Target Encoding**: Selalu gunakan data training saja untuk menghitung statistics
2. **Interaction Features**: Coba kombinasi fitur yang logically related
3. **Binning**: Untuk fitur continuous dengan non-linear relationship
4. **NLP**: TF-IDF + SVD efektif untuk text features
5. **Domain Knowledge**: Pahami domain musik untuk bikin fitur meaningful

---

## 🔄 Iterasi Future

Fitur tambahan yang bisa dicoba:

1. **Artist momentum**: Tren popularitas artist (naik/turun)
2. **Genre statistics**: Stats per genre (mean, std, percentile)
3. **Audio clusters**: K-means clustering pada audio features
4. **Lyrics complexity**: Readability score, unique words count
5. **Seasonal effects**: Bulan rilis (summer hits vs winter songs)
6. **Collaboration features**: Solo vs featuring, jumlah collaborators
7. **Cross-genre features**: Hybrid genre detection

---

**Catatan**: Semua fitur di atas dirancang untuk menangkap berbagai aspek yang mempengaruhi popularitas lagu, dari karakteristik artist, kualitas audio, hingga konten lyrics dan temporal trends.
