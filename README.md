# Academic Projects Collection

This repository contains my academic and self-learning projects, focusing on **Data Structures & Algorithms visualization** and **Data Science with Python**.

## Table of Contents

- [Projects](#projects)
  - [1. Sorting Algorithm Visualizer (Web)](#1-sorting-algorithm-visualizer-web)
  - [2. Feed Grains Exploratory Data Analysis (Python)](#2-feed-grains-exploratory-data-analysis-python)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contact](#contact)

---

## Projects

### 1. Sorting Algorithm Visualizer (Web)

**Goal:**  
An interactive web-based tool to help users clearly understand how different sorting algorithms work step-by-step.

**Description:**

- Visualizes popular sorting algorithms:
  - Bubble Sort  
  - Selection Sort  
  - Insertion Sort  
  - Quick Sort  
  - Merge Sort
- Animates comparisons and swaps in real time.
- Allows:
  - Dynamic array generation (random values)
  - Control over animation speed
  - Control over array size / number of elements
- Highlights:
  - Elements being compared
  - Elements being swapped
  - Sorted portion of the array

**Tech Stack:**

- HTML, CSS, JavaScript
- DOM manipulation for bar visualization
- Simple, responsive UI focused on learning and clarity

**Usage (Web Project):**

1. Open `index.html` in any modern browser.
2. Use the controls to:
   - Generate a new array
   - Select an algorithm
   - Adjust speed / size
   - Start the visualization

---

### 2. Feed Grains Exploratory Data Analysis (Python)

**Goal:**  
To perform an in-depth exploratory data analysis (EDA) on historical U.S. feed grain data (corn, barley, oats, sorghum, soybean meal, etc.) and derive insights for agricultural planning and policy. :contentReference[oaicite:0]{index=0}  

**Dataset:**

- Source: USDA Feed Grains Database (Economic Research Service)
- File: `FeedGrains.xls`
- Contains:
  - Yield per harvested acre
  - Prices received by farmers
  - Ending stocks
  - Exports
  - Multiple commodities across several decades

**Key Steps & Analyses:**

- **Data Cleaning**
  - Handled missing and incorrect values (e.g., `-1`, placeholders)
  - Removed unnecessary rows/columns
  - Converted data types for numerical analysis

- **Exploratory Data Analysis**
  - Distribution analysis of yields and prices (histograms, box/boxen plots)
  - Time-series trends:
    - Yield trends over years
    - Median price trends per commodity
    - Rolling averages to smooth volatility
  - Commodity-level insights:
    - Top commodities by number of data points
    - Yield distribution by commodity
  - Statistical analysis:
    - Correlation between metrics (yield, price, expo
