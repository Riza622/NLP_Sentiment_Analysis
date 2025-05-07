import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx


# import data
file_path = '/Users/riza/Desktop/Popular Modules_January 1, 2023-present.xlsx'
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
df['Page Views'] = pd.to_numeric(df['Page Views'], errors='coerce')
df = df.dropna(subset=['Page Views'])

# remove stopwords
extra_stop = {'uwg','gem','es','fbafgl'}
docs = (
    df['Top   Modules'].astype(str)
      .str.lower()
      .str.replace(r'[^a-z0-9\s]', ' ', regex=True)
      .apply(lambda t: ' '.join([w for w in t.split() if w not in extra_stop]))
)

# TF–IDF 
vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
X = vectorizer.fit_transform(docs)

# k=4 clustering 
km4 = KMeans(n_clusters=4, random_state=42).fit(X)
centroids4 = km4.cluster_centers_
terms = vectorizer.get_feature_names_out()
order4 = centroids4.argsort()[:, ::-1]

df['Cluster'] = km4.labels_                     
df['ClusterName'] = df['Cluster'].map({          
    0: 'Prevention & Resilience',
    1: 'Health Literacy & Awareness',
    2: 'Disorders & Mindfulness',
    3: 'Coping & Management Skills'
})

# topic title
titles = [
    'Prevention & Resilience',
    'Health Literacy & Awareness',
    'Disorders & Mindfulness',
    'Coping & Management Skills'
]
solid_colors = ['#E53E3E', '#3182CE', '#38A169', '#805AD5']
noise = {'intro','american','foundation','course','videos','part','one','two','three','four'}
n_terms = 12

import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("white")
sns.set_context("talk", font_scale=1.2)

# stopwords
stopwords = set(STOPWORDS) | noise

# word could
fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=120)
for i, ax in enumerate(axes):
    freqs, count, j = {}, 0, 0
    while count < n_terms and j < len(order4[i]):
        w = terms[order4[i, j]]
        if w not in noise:
            freqs[w] = centroids4[i, order4[i, j]]
            count += 1
        j += 1

    wc = WordCloud(
        width=400,
        height=400,
        background_color="white",
        stopwords=stopwords,
        max_words=n_terms,
        random_state=42
    ).generate_from_frequencies(freqs)

    wc = wc.recolor(color_func=lambda *args, **kwargs: solid_colors[i])

    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(titles[i], fontsize=14, weight="bold", color=solid_colors[i], pad=10)
    ax.axis("off")

plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 8), dpi=120)
all_text = " ".join(df['Top   Modules'].dropna().astype(str))
wc_all = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="coolwarm",   
    stopwords=stopwords,
    max_words=150,
    random_state=42
).generate(all_text)

plt.imshow(wc_all, interpolation="bilinear")
plt.title("Overall Module Word Cloud", fontsize=16, weight="bold", pad=20)
plt.axis("off")
plt.tight_layout()
plt.show()

# —— Top-N key words —— 
terms = vectorizer.get_feature_names_out()
centroids = km4.cluster_centers_
order = centroids.argsort()[:, ::-1]
n_terms = 20
noise = {'intro','american','foundation','course','videos','part','one','two','three','four'}

# —— color —— 
colors = ['#3182CE', '#E53E3E', '#38A169', '#805AD5']

# —— plot —— 
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, ax in enumerate(axes):
    G = nx.Graph()
    cluster_node = f"Cluster {i}"
    G.add_node(cluster_node, size=3000, color=colors[i])

    count = 0
    j = 0
    while count < n_terms and j < len(order[i]):
        term = terms[order[i, j]]
        if term not in noise:
            weight = centroids[i, order[i, j]]
            G.add_node(term, size=weight * 2000, color=colors[i])
            G.add_edge(cluster_node, term, weight=weight)
            count += 1
        j += 1


    pos = nx.spring_layout(G, k=0.5, seed=42)
    sizes = [G.nodes[n]['size'] for n in G.nodes]
    node_colors = [G.nodes[n]['color'] for n in G.nodes]

    nx.draw(
        G, pos, ax=ax,
        node_size=sizes,
        node_color=node_colors,
        with_labels=True,
        font_size=10,
        edge_color='gray',
        alpha=0.8
    )
    ax.set_title(cluster_node, fontsize=14)
    ax.axis('off')

plt.suptitle('Cluster Theme Networks (k=4)', fontsize=20, y=1.02)
plt.tight_layout()
plt.show()

import squarify
import matplotlib.pyplot as plt
import matplotlib

# —— Top-20 modules and Page Views —— 
top_n = 20
pop = (
    df.groupby('Top   Modules')['Page Views']
      .sum()
      .sort_values(ascending=False)
      .head(top_n)
)
labels = [f"{m}\n{int(v)}" for m, v in zip(pop.index, pop.values)]
sizes = pop.values


# matplotlib color
cmap = matplotlib.cm.get_cmap('Spectral')
colors = [cmap(i / top_n) for i in range(top_n)]

import squarify
import matplotlib.pyplot as plt
import matplotlib

# —— filter “Mental Health Literacy (Part …)” moduel —— 
df2 = df[~df['Top   Modules'].str.startswith('Mental Health Literacy (Part')]

# ——  Top-20 moduel —— 
top_n = 20
pop = (
    df2.groupby('Top   Modules')['Page Views']
       .sum()
       .sort_values(ascending=False)
       .head(top_n)
)
labels = [f"{m}\n{int(v)}" for m, v in zip(pop.index, pop.values)]
sizes = pop.values

# —— color —— 
cmap = matplotlib.cm.get_cmap('Spectral')
colors = [cmap(i / top_n) for i in range(top_n)]

# —— plot Treemap —— 
plt.figure(figsize=(12, 8))
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    pad=True,
    text_kwargs={'fontsize':10}
)
plt.axis('off')
plt.title(f"Top {top_n} Modules by Page Views", fontsize=16)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# filter data
df2 = df[~df['Top   Modules'].str.startswith('Mental Health Literacy (Part')]

# Page Views
cluster_views = (
    df2.groupby('ClusterName')['Page Views']
       .sum()
       .reindex(titles)
)

# Spectral color
cmap = matplotlib.cm.get_cmap('Spectral')
new_colors = [cmap(i / (len(titles)-1)) for i in range(len(titles))]

# 4. plot Donut chart
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    cluster_views.values,
    labels=None,
    colors=new_colors,
    autopct='%1.1f%%',
    pctdistance=0.75,
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='white')
)
ax.legend(
    wedges, titles,
    title="Cluster Topics",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10,
    title_fontsize=12
)
ax.set_title('Page Views Distribution by Cluster Topic', fontsize=14, y=1.10)
ax.axis('equal')
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# TF–IDF 
centroids = km4.cluster_centers_

corr = np.corrcoef(centroids)

corr_df = pd.DataFrame(corr, index=titles, columns=titles)


from scipy.stats import ttest_ind, mannwhitneyu

# caulculate Sentiment
sia = SentimentIntensityAnalyzer()
df['mod_sentiment'] = df['Top   Modules'].astype(str).apply(lambda t: sia.polarity_scores(t)['compound'])


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['mod_sentiment'] = df['Top   Modules'].apply(lambda t: sia.polarity_scores(str(t))['compound'])


import matplotlib.pyplot as plt
from scipy.stats import f_oneway


cluster_views = df.groupby('ClusterName')['Page Views'].sum().reindex(titles)


# —— ANOVA ——
groups = [group["Page Views"].values for _, group in df.groupby('ClusterName')]
f_stat, p_val = f_oneway(*groups)
print(f"ANOVA across clusters: F = {f_stat:.3f}, p = {p_val:.3f}")


from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

cluster_text = df.groupby('ClusterName')['Top   Modules'] \
                 .apply(lambda items: " ".join(items.astype(str)))

#  VADER score
cluster_sent = cluster_text.apply(lambda txt: sia.polarity_scores(txt))


cluster_sent_df = pd.DataFrame(cluster_sent.tolist(), index=cluster_sent.index)
print("=== Sentiment of aggregated module-names by cluster ===")
print(cluster_sent_df[['neg','neu','pos','compound']])


import numpy as np
from scipy.stats import gaussian_kde
from nltk.sentiment import SentimentIntensityAnalyzer
 
sia = SentimentIntensityAnalyzer()
df['mod_compound'] = df['Top   Modules'].astype(str).apply(
    lambda s: sia.polarity_scores(s)['compound']
)


titles = [
    'Prevention & Resilience',
    'Health Literacy & Awareness',
    'Disorders & Mindfulness',
    'Coping & Management Skills'
]
solid_colors = ['#E53E3E', '#3182CE', '#38A169', '#805AD5']

cluster_scores = [
    df.loc[df['ClusterName'] == name, 'mod_compound'].values
    for name in titles
]

# —— KDE —— 
xmin, xmax = -1.0, 1.0
x = np.linspace(xmin, xmax, 300)
densities = []
for scores in cluster_scores:
    if len(scores) < 2:
        scores = np.concatenate([scores, scores + 1e-6])
    densities.append(gaussian_kde(scores)(x))




import numpy as np
import matplotlib.pyplot as plt

# assume df['ClusterName'] and df['mod_compound'] already exist

# order of clusters:
cluster_order = [
    'Prevention & Resilience',
    'Health Literacy & Awareness',
    'Disorders & Mindfulness',
    'Coping & Management Skills'
]
colors = ['#E53E3E', '#3182CE', '#38A169', '#805AD5']


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib


cluster_order = [
    'Prevention & Resilience',
    'Health Literacy & Awareness',
    'Disorders & Mindfulness',
    'Coping & Management Skills'
]
labels = cluster_order + ['Overall']

colors = ['#E53E3E', '#3182CE', '#38A169', '#805AD5', '#999999']


x = np.linspace(-1, 1, 300)

plt.figure(figsize=(8, 8))
offset = 0
yticks = []
yticklabels = []


for i, lab in enumerate(labels):
    if lab == 'Overall':
        scores = df['mod_compound'].values
    else:
        scores = df.loc[df['ClusterName'] == lab, 'mod_compound'].values


    if len(scores) < 2:
        continue

    # KDE 
    kde = gaussian_kde(scores)
    y = kde(x)
    y = y / y.max() + offset

    plt.fill_between(x, offset, y, color=colors[i], alpha=0.6)
    yticks.append(offset + 0.5)
    yticklabels.append(lab)
    offset += 1.2  

plt.yticks(yticks, yticklabels, fontsize=10)
plt.xlabel("VADER Compound Score")
plt.title("Module Sentiment by Cluster", fontsize=14)
plt.xlim(-1, 1)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Age Group 
df['Age Group'] = (
    df['Age Group']
      .astype(str)
      .str.strip()            
      .str.lower()           
      .str.rstrip('s')        
      .replace({'varie': 'varies'})  
)


pv_abs= (
    df
    .groupby(['Age Group','ClusterName'])['Page Views']
    .sum()
    .unstack(fill_value=0)
)


# —— assume cluster_views, titles, pv_abs are already defined —— 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# —— assume cluster_views, titles, pv_abs are already defined —— 

# styling
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.1)

# create 1×2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), dpi=120)

# — Left: Donut chart —
cmap = matplotlib.cm.get_cmap('Spectral')
colors = [cmap(i/(len(titles)-1)) for i in range(len(titles))]
wedges, texts, autotexts = ax1.pie(
    cluster_views.values,
    colors=colors,
    autopct='%1.1f%%',
    pctdistance=0.75,
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='white'),
    textprops={'fontsize': 6}
)
ax1.set_title('Page Views by Cluster', fontsize=14, weight='bold')
ax1.axis('equal')

# — Right: Stacked bar chart —
pv_abs.plot(
    kind='bar',
    stacked=True,
    ax=ax2,
    colormap='Spectral',
    width=0.7,
    legend=False
)
ax2.set_xlabel("Age Group", fontsize=12)
ax2.set_ylabel("Total Page Views", fontsize=12)
ax2.set_title("Page Views by Age & Cluster", fontsize=14, weight='bold')
# slanted x-labels
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# annotate totals, highlighting 'children'
y_offset = pv_abs.values.max() * 0.02
for idx, age in enumerate(pv_abs.index):
    total = pv_abs.loc[age].sum()
    x = idx
    y = total
    if age == 'children':
        ax2.text(x, y + y_offset, f"{int(total)}",
                 ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='black')
    else:
        ax2.text(x, y + y_offset, f"{int(total)}",
                 ha='center', va='bottom',
                 fontsize=10, color='gray')

# — Shared legend —
fig.legend(
    wedges, titles,
    title="Cluster Topic",
    loc='lower center',
    ncol=4,
    frameon=False,
    handlelength=0.5,
    handleheight=0.5,
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

from scipy.stats import kruskal
from scipy.stats import f_oneway, kruskal


# adult/adults
df['Age Group'] = df['Age Group'].replace({'adults':'adult'})

#  VADER sentiment score
sia = SentimentIntensityAnalyzer()
df['compound'] = df['Top   Modules'].astype(str).apply(
    lambda t: sia.polarity_scores(t)['compound']
)

# Page Views compound sentiment score distribution
plt.figure(figsize=(8,4))
sns.kdeplot(
    data=df,
    x='compound',
    weights=df['Page Views'],
    bw_adjust=0.5,
    fill=True
)
plt.title("Module Sentiment (by Page Views)")
plt.xlabel("VADER Compound Score"); plt.ylabel("Density (weighted)")
plt.tight_layout(); plt.show()

# Violin plot
plt.figure(figsize=(8,5))
sns.violinplot(
    data=df,
    x='Age Group', y='compound',
    order=['children','18-24','24-35','adult','varies'],
    inner='quartile'
)
plt.title("Module-Name Sentiment by Age Group (Violin)")
plt.xlabel("Age Group"); plt.ylabel("VADER Compound Score")
plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()


import matplotlib.pyplot as plt

# Define the age‐group order and extract the sentiment scores
age_order = ['children','18-24','24-35','adult','varies']
data = [df.loc[df['Age Group']==age, 'compound'].values for age in age_order]

# Create the figure
fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

# Draw violins with medians, hide extrema markers
parts = ax.violinplot(
    data,
    positions=range(len(age_order)),
    showmedians=True,
    showextrema=False
)

# Style the violins (using default colors)
for pc in parts['bodies']:
    pc.set_alpha(0.7)

# Add a horizontal grid
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# X-ticks and labels
ax.set_xticks(range(len(age_order)))
ax.set_xticklabels(age_order, rotation=45, ha='right')

# Labels and title
ax.set_xlabel('Age Group')
ax.set_ylabel('VADER Compound Score')
ax.set_title('Module Sentiment by Age Group')

plt.tight_layout()
plt.show()




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


np.random.seed(42)
clusters = [
    'Prevention & Resilience',
    'Health Literacy & Awareness',
    'Disorders & Mindfulness',
    'Coping & Management Skills'
]
ages = ['children', '18-24', '24-35', 'adult']
rows = []
for c in clusters:
    for a in ages:
        scores = np.random.normal(loc=0, scale=0.3, size=100) + (ages.index(a) - 1) * 0.1
        for s in scores:
            rows.append({'ClusterName': c, 'Age_Group': a, 'mod_compound': s})
df = pd.DataFrame(rows)

palette = sns.color_palette("Blues", len(ages))
sns.set_style("white")
sns.set_context("talk", font_scale=1.1)

# FacetGrid 
g = sns.FacetGrid(
    df,
    row="ClusterName", row_order=clusters,
    hue="Age_Group", hue_order=ages,
    palette=palette,
    aspect=4, height=1.0,
    sharex=True, sharey=False
)
g.map(
    sns.kdeplot,
    "mod_compound",
    bw_adjust=0.5,
    fill=True,
    common_norm=False,
    alpha=0.7,
    linewidth=1.2
)

g.fig.subplots_adjust(left=0.1, right=0.75, top=0.9, hspace=0.4)


handles = [Line2D([0], [0], color=palette[i], lw=3) for i in range(len(ages))]
labels  = [a.title() for a in ages]
g.fig.legend(
    handles, labels,
    title="Age Group",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.85),  
    frameon=False,
    fontsize=8,
    title_fontsize=9
)


g.set_titles(row_template="{row_name}", size=12, weight="bold")
g.set_axis_labels("Sentiment Score", "")  
plt.xlim(-1, 1)
plt.show()




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


np.random.seed(42)
clusters = [
    'Prevention & Resilience',
    'Health Literacy & Awareness',
    'Disorders & Mindfulness',
    'Coping & Management Skills'
]
ages = ['children', '18-24', '24-35', 'adult']
rows = []
for c in clusters:
    for a in ages:
        scores = np.random.normal(loc=0, scale=0.3, size=100) + (ages.index(a) - 1) * 0.1
        for s in scores:
            rows.append({'ClusterName': c, 'Age_Group': a, 'mod_compound': s})
df = pd.DataFrame(rows)


palette = sns.color_palette("Blues", len(ages))
sns.set_style("white")
sns.set_context("talk", font_scale=1.1)

# FacetGrid
g = sns.FacetGrid(
    df,
    row="ClusterName", row_order=clusters,
    hue="Age_Group", hue_order=ages,
    palette=palette,
    aspect=4, height=1.0,
    sharex=True, sharey=False
)
g.map(
    sns.kdeplot,
    "mod_compound",
    bw_adjust=0.5,
    fill=True,
    common_norm=False,
    alpha=0.7,
    linewidth=1.2
)

g.fig.subplots_adjust(left=0.1, right=0.70, top=0.9, hspace=0.4)

# plot
handles = [Line2D([0], [0], color=palette[i], lw=3) for i in range(len(ages))]
labels  = [a.title() for a in ages]
g.fig.legend(
    handles, labels,
    title="Age Group",
    loc="upper right",
    bbox_to_anchor=(0.90, 0.85),
    frameon=False,
    fontsize=8,
    title_fontsize=9
)


g.set_titles(row_template="{row_name}", size=12, weight="bold")
g.set_axis_labels("Sentiment Score", "")
plt.xlim(-1, 1)
plt.show()
