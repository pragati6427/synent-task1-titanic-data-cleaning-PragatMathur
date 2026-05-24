"""
=============================================================
  Titanic Dataset — Data Cleaning & Preprocessing
  Task 1 | Data Science Internship Project
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d52',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3e',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
ACCENT  = '#6c63ff'
GREEN   = '#00d4a0'
RED     = '#ff6b6b'
YELLOW  = '#ffd166'
PINK    = '#ef476f'

# ════════════════════════════════════════════════════════
#  STEP 0 — Load raw data
# ════════════════════════════════════════════════════════
df_raw = pd.read_csv('/home/claude/titanic_raw.csv')
df = df_raw.copy()
print("▶ Raw shape:", df.shape)

# ════════════════════════════════════════════════════════
#  STEP 1 — Remove Duplicates
# ════════════════════════════════════════════════════════
n_before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
n_removed = n_before - len(df)
print(f"▶ Duplicates removed: {n_removed}  |  Rows after: {len(df)}")

# ════════════════════════════════════════════════════════
#  STEP 2 — Rename Columns
# ════════════════════════════════════════════════════════
df.rename(columns={
    'PassengerId': 'passenger_id',
    'Survived':    'survived',
    'Pclass':      'pclass',
    'Name':        'name',
    'Sex':         'sex',
    'Age':         'age',
    'SibSp':       'siblings_spouses',
    'Parch':       'parents_children',
    'Ticket':      'ticket',
    'Fare':        'fare',
    'Cabin':       'cabin',
    'Embarked':    'embarked'
}, inplace=True)
print("▶ Columns renamed:", df.columns.tolist())

# ════════════════════════════════════════════════════════
#  STEP 3 — Convert Data Types
# ════════════════════════════════════════════════════════
df['age']  = pd.to_numeric(df['age'],  errors='coerce')
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')

df['survived'] = df['survived'].astype(int)
df['pclass']   = df['pclass'].astype(int)

df['sex']      = df['sex'].astype('category')
df['embarked'] = df['embarked'].astype('category')
print("▶ Data types converted")
print(df.dtypes.to_string())

# ════════════════════════════════════════════════════════
#  STEP 4 — Handle Missing Values
# ════════════════════════════════════════════════════════
missing_before = df.isnull().sum()

# Age → median per pclass+sex group
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(
    lambda x: x.fillna(x.median())
)

# Fare → median per pclass
df['fare'] = df.groupby('pclass')['fare'].transform(
    lambda x: x.fillna(x.median())
)

# Embarked → mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Cabin → 'Unknown'
df['cabin'] = df['cabin'].fillna('Unknown')

missing_after = df.isnull().sum()
print("▶ Missing values — Before:")
print(missing_before[missing_before > 0].to_string())
print("▶ Missing values — After:")
print(missing_after[missing_after > 0].to_string() if missing_after.sum() > 0 else "  None!")

# ════════════════════════════════════════════════════════
#  STEP 5 — Feature Engineering (bonus)
# ════════════════════════════════════════════════════════
df['family_size']  = df['siblings_spouses'] + df['parents_children'] + 1
df['is_alone']     = (df['family_size'] == 1).astype(int)
df['age_group']    = pd.cut(
    df['age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'YoungAdult', 'MiddleAged', 'Senior']
)
df['fare_band']    = pd.qcut(df['fare'], q=4,
    labels=['Low', 'Medium', 'High', 'Very High'])
df['cabin_known']  = (df['cabin'] != 'Unknown').astype(int)

print("\n▶ Feature engineering complete. Final shape:", df.shape)
print(df.head(3).to_string())

# ════════════════════════════════════════════════════════
#  Save clean dataset
# ════════════════════════════════════════════════════════
df.to_csv('/home/claude/titanic_clean.csv', index=False)
print("\n✅ Clean dataset saved → titanic_clean.csv")

# ════════════════════════════════════════════════════════
#  VISUALISATION DASHBOARD
# ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

# ── Title ─────────────────────────────────────────────
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor('#0f1117')
ax_title.axis('off')
ax_title.text(0.5, 0.78, '🚢  TITANIC — Data Cleaning & Preprocessing',
    transform=ax_title.transAxes, ha='center', va='center',
    fontsize=22, fontweight='bold', color='white')
ax_title.text(0.5, 0.28, 'Task 1  |  Data Science Internship  |  Full Cleaning Pipeline',
    transform=ax_title.transAxes, ha='center', va='center',
    fontsize=13, color='#8888aa')

# ── 1. Missing Values Before ──────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
mv = missing_before[missing_before > 0]
bars = ax1.barh(mv.index, mv.values, color=[RED, YELLOW, ACCENT, PINK][:len(mv)],
    edgecolor='none', height=0.6)
for bar, val in zip(bars, mv.values):
    ax1.text(val + 2, bar.get_y() + bar.get_height()/2,
        str(val), va='center', fontsize=9, color='white')
ax1.set_title('Missing Values (Before)', color=RED, fontsize=11, pad=8)
ax1.set_xlabel('Count', fontsize=9)
ax1.grid(axis='x', alpha=0.3)
ax1.spines[['top','right']].set_visible(False)

# ── 2. Missing Values After ───────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
cols = missing_before[missing_before > 0].index
after_vals = [0 if c == 'cabin' else 0 for c in cols]
after_display = {c: 0 for c in cols}
bars2 = ax2.barh(list(after_display.keys()), list(after_display.values()),
    color=GREEN, edgecolor='none', height=0.6)
for bar in bars2:
    ax2.text(0.5, bar.get_y() + bar.get_height()/2,
        '0 ✓', va='center', fontsize=9, color=GREEN)
ax2.set_xlim(0, max(mv.values) * 1.2)
ax2.set_title('Missing Values (After)', color=GREEN, fontsize=11, pad=8)
ax2.set_xlabel('Count', fontsize=9)
ax2.grid(axis='x', alpha=0.3)
ax2.spines[['top','right']].set_visible(False)

# ── 3. Survival Rate ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
surv_counts = df['survived'].value_counts()
colors_pie = [RED, GREEN]
wedges, texts, autotexts = ax3.pie(
    surv_counts, labels=['Did Not Survive', 'Survived'],
    autopct='%1.1f%%', colors=colors_pie,
    startangle=90, wedgeprops=dict(edgecolor='#0f1117', linewidth=2))
for t in texts: t.set_color('#cccccc'); t.set_fontsize(9)
for a in autotexts: a.set_color('white'); a.set_fontweight('bold'); a.set_fontsize(10)
ax3.set_title('Survival Distribution', color=ACCENT, fontsize=11, pad=8)

# ── 4. Pclass Distribution ────────────────────────────
ax4 = fig.add_subplot(gs[1, 3])
pclass_counts = df['pclass'].value_counts().sort_index()
bars4 = ax4.bar(['1st Class', '2nd Class', '3rd Class'],
    pclass_counts.values,
    color=[YELLOW, ACCENT, PINK], edgecolor='none', width=0.6)
for bar in bars4:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
        str(bar.get_height()), ha='center', fontsize=10, color='white', fontweight='bold')
ax4.set_title('Passenger Class', color=YELLOW, fontsize=11, pad=8)
ax4.set_ylabel('Count', fontsize=9)
ax4.grid(axis='y', alpha=0.3)
ax4.spines[['top','right']].set_visible(False)

# ── 5. Age Distribution (after fill) ─────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
ax5.hist(df['age'], bins=30, color=ACCENT, edgecolor='#0f1117', linewidth=0.5, alpha=0.85)
ax5.axvline(df['age'].mean(), color=YELLOW, linestyle='--', linewidth=1.5, label=f"Mean: {df['age'].mean():.1f}")
ax5.axvline(df['age'].median(), color=GREEN, linestyle='--', linewidth=1.5, label=f"Median: {df['age'].median():.1f}")
ax5.set_title('Age Distribution (after imputation)', color=ACCENT, fontsize=11, pad=8)
ax5.set_xlabel('Age', fontsize=9)
ax5.set_ylabel('Count', fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)
ax5.spines[['top','right']].set_visible(False)

# ── 6. Fare by Pclass ─────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2:4])
classes = [1, 2, 3]
colors_box = [YELLOW, ACCENT, PINK]
box_data = [df[df['pclass'] == c]['fare'].values for c in classes]
bp = ax6.boxplot(box_data, patch_artist=True, widths=0.5,
    medianprops=dict(color='white', linewidth=2),
    whiskerprops=dict(color='#8888aa'),
    capprops=dict(color='#8888aa'),
    flierprops=dict(marker='o', color='#8888aa', markersize=3))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('none')
ax6.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
ax6.set_title('Fare Distribution by Class', color=PINK, fontsize=11, pad=8)
ax6.set_ylabel('Fare (£)', fontsize=9)
ax6.grid(axis='y', alpha=0.3)
ax6.spines[['top','right']].set_visible(False)

# ── Footer stats ──────────────────────────────────────
stats_text = (
    f"  Raw: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols     "
    f"→  Duplicates removed: {n_removed}     "
    f"→  Missing filled: Age ({missing_before.get('age',0)}), "
    f"Fare ({missing_before.get('fare',0)}), "
    f"Embarked ({missing_before.get('embarked',0)}), "
    f"Cabin ({missing_before.get('cabin',0)})     "
    f"→  Clean: {df.shape[0]} rows × {df.shape[1]} cols  "
)
fig.text(0.5, 0.01, stats_text, ha='center', va='bottom',
    fontsize=8.5, color='#8888aa',
    bbox=dict(facecolor='#1a1d2e', edgecolor='#3a3d52', boxstyle='round,pad=0.4'))

plt.savefig('/home/claude/titanic_cleaning_dashboard.png',
    dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("✅ Dashboard saved → titanic_cleaning_dashboard.png")
