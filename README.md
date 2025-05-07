# NLP Sentiment Analysis of Popular Modules

A streamlined pipeline to cluster “Top Modules” text, visualize thematic word clouds and network graphs, and analyze module-name sentiment across clusters and age groups.

---

- **Data Ingestion & Cleaning**  
  Load Excel data, normalize columns, filter page‐view counts, remove custom stopwords.

- **Text Vectorization & Clustering**  
  TF–IDF transformation + K-Means (k=4) → four topic clusters:
  1. Prevention & Resilience  
  2. Health Literacy & Awareness  
  3. Disorders & Mindfulness  
  4. Coping & Management Skills  

- **Visualizations**  
  - Cluster‐specific word clouds  
  - Overall word cloud  
  - Network graphs of top keywords  
  - Treemap & donut charts of page‐view distributions  
  - Correlation heatmap of cluster centroids  
  - KDE & violin plots of VADER sentiment scores by cluster & age  
  - Weighted sentiment density plots  

- **Statistical Analysis**  
  - VADER sentiment scoring  
  - ANOVA & nonparametric tests for cluster differences  

---

