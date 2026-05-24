# 🚢 Titanic — Data Science Internship Tasks

## Task 1: Data Cleaning & Preprocessing

**Dataset:** Titanic (891 passengers)  
**Tools:** Python, Pandas, NumPy, Matplotlib

### What was done
- Removed 3 duplicate rows
- Renamed columns to snake_case
- Converted data types (age/fare → float64, sex/embarked → category)
- Handled missing values (Age: median by class+sex, Fare: median by class, Embarked: mode, Cabin: 'Unknown')
- Engineered 5 new features: family_size, is_alone, age_group, fare_band, cabin_known

### Task 2: EDA — Survival Analysis
- Survival rate by sex, passenger class, and age group
- Heatmap: sex × class survival rates

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python titanic_cleaning.py
```

## Demo video
[Link to your video here]
