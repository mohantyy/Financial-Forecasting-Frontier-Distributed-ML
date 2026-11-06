# Financial Forecasting Frontier: Distributed Machine Learning in Banking
**Author:** Abhilash Mohanty (@mohantyy)
**Program:** AlmaBetter Masters in Data Science — ML Engineering v1 Capstone Project
**Dataset:** UCI Bank Marketing Dataset (`bank.csv`, 4,521 rows x 17 attributes)

## 1. Executive Summary & 5-Part Distributed Architecture
- **Part 1 (Hadoop HDFS, MapReduce & Apache Hive):** Ingested `bank.csv` (4,521 rows, 0 nulls) into HDFS and executed 4 Python MapReduce streaming jobs and 9 HiveQL query suites (`banking_data.client_info`). Identified `retired` (€2,528.84) and `management` (€2,143.40) as top-balance professions, 449 married personal-loan holders, and 85.03% conversion for prior campaign `success`.
- **Part 2 (Exploratory Data Analysis with Apache Spark & Spark SQL):** Executed 12 PySpark DataFrame transformations, custom `@udf` age categorization (`<30`, `30-60`, `>60`), temporary SQL views (`bank_view`), and 15 UBM visualizations. Confirmed 1,839 (40.68%) clients have balance > €1,000 and `cellular` outreach achieves 34.33% conversion vs 9.11% for `unknown`.
- **Part 3 (Predictive Modeling with Spark MLlib):** Built a 54-feature `VectorAssembler` + `StandardScaler` pipeline with training-only SMOTE balancing (5,332 balanced train rows) and 5-fold `CrossValidator` tuning. Champion Tuned `GBTClassifier` achieved **96.69% Accuracy**, **94.07% Precision**, **93.28% Recall**, **0.9367 F1-Score**, and **0.9911 ROC-AUC** on the 905-row test set.
- **Part 4 (Data Parallelism & Resource Optimization):** Partitioned data across 8 hash-balanced workers by `job`, achieving a **5.41x speedup** (76.2 ms vs 412.6 ms single partition, 59,331 rows/sec) and identifying the 40-49 (€20.12M) and 30-39 (€19.05M) age cohorts as the highest loan-exposure segments.
- **Part 5 (Real-Time Transaction Analysis with Spark Structured Streaming):** Streamed 10 micro-batches (`4,521` events) with real-time `GBTClassifier` scoring across `10s` and `1m` tumbling windows and a **30-second event-time watermark (`withWatermark`)**, reconciling 117 late events and dropping 62 expired records.

## 2. Spark MLlib Model Benchmark (n = 905 Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 94.59% | 85.93% | 94.96% | 0.9022 | 0.9866 |
| Logistic Regression (Tuned + SMOTE) | 96.24% | 92.50% | 93.28% | 0.9289 | 0.9899 |
| Random Forest (Tuned, 60 Trees) | 95.14% | 89.43% | 92.44% | 0.9091 | 0.9868 |
| **GBTClassifier (Champion Tuned)** | **96.69%** | **94.07%** | **93.28%** | **0.9367** | **0.9911** |
