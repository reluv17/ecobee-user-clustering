# Ecobee Thermostat User Clustering — Unsupervised Learning

Identifying natural behavioral segments among 1,000 Ecobee smart thermostat users using K-Means, Gaussian Mixture Models, and DBSCAN. This project covers user-level feature engineering from 8.5M hourly records, anomaly detection, density estimation, and product/marketing recommendations based on discovered segments.

---

## Overview

Understanding *how* users interact with their smart thermostats — not just whether overrides happen — enables personalized product development and targeted marketing. This project aggregates raw thermostat data into 48 behavioral features per user, then applies multiple clustering algorithms to discover meaningful user segments with distinct comfort preferences and usage patterns.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Ecobee smart thermostat network |
| Period | April 1 – September 30, 2017 |
| Raw records | 8,580,538 hourly observations |
| Users | 1,000 |
| Override rate | 29.1% across all users |

**Raw features:** Event type, schedule type, indoor temperature, indoor humidity, outdoor temperature, outdoor humidity, cool setpoint, heat setpoint, override status

---

## Feature Engineering

Each user was summarized into 48 behavioral features aggregated from their full 6-month history:

**Comfort metrics:** % time in deadband, % time too cold/hot, avg temperature relative to setpoints, deadband width mean and variability

**Setpoint behavior:** Heat/cool setpoint means and standard deviations, weekday vs weekend setpoints, setpoints during extreme weather, seasonal setpoint changes

**Schedule adherence:** Override rate, weekday override rate, heat/cool changes per day, schedule change frequency

**Occupancy patterns:** % time away vs home, away/home setpoint differentials, Smart Away usage

**Temporal patterns:** Morning/evening peak behavior, hour-of-day variance, weekday-weekend behavioral differences

6 highly correlated features (|r| > 0.9) were removed before clustering to reduce redundancy. Missing values filled with training medians.

---

## Clustering Methods

Three algorithms were evaluated across multiple hyperparameter settings:

### K-Means
- Tested k = 2 through 10
- Evaluated via Silhouette score, Davies-Bouldin index, Calinski-Harabasz index, and elbow method
- **Optimal: k = 3** (Silhouette = 0.136, best interpretability)

### Gaussian Mixture Models (GMM)
- Tested k = 2 through 10
- Evaluated via BIC, AIC, and Silhouette
- BIC-optimal: k = 3 — consistent with K-Means

### DBSCAN
- Tested multiple eps/min_samples combinations
- Best configuration (eps=4.0, min_samples=5): 2 clusters, 42.6% noise
- High noise rate indicates data does not have strong density-based structure

**Selected model: K-Means k=3** — highest interpretability, consistent across metrics, meaningful cluster sizes

---

## Advanced Analysis

**Anomaly Detection (Local Outlier Factor):** 100 anomalous users (10%) identified — characterized by significantly higher override rates (+31%) and wider deadbands (+23%) than typical users

**Covariance Estimation:** Empirical covariance reveals all three clusters have elongated/elliptical shapes — suggesting K-Means (which assumes spherical clusters) captures directional behavioral patterns

**Kernel Density Estimation:** Cross-validated bandwidth = 0.800; density peaks align with the three K-Means clusters, validating their structure

---

## Cluster Profiles

### Cluster 0 — Energy Savers (229 users, 22.9%)
| Feature | Value |
|---|---|
| Override rate | 15.6% |
| Deadband width | 13.8°F |
| Time in deadband | 75.4% |
| Away-home heat differential | 3.5°F |

Wide comfort tolerance, strong away-mode usage, low override rate. These users actively manage energy costs and are comfortable with a wide temperature range.

---

### Cluster 1 — Active Controllers (172 users, 17.2%)
| Feature | Value |
|---|---|
| Override rate | 78.7% |
| Deadband width | 3.4°F |
| Time in deadband | 32.4% |
| Away-home heat differential | 0.1°F |

Narrow comfort zone, very high override rate, minimal away-mode differentiation. These users frequently override the schedule and spend only 32% of time within their own setpoint range — suggesting the thermostat has not learned their preferences.

---

### Cluster 2 — Balanced Users (599 users, 59.9%)
| Feature | Value |
|---|---|
| Override rate | 19.8% |
| Deadband width | 8.3°F |
| Time in deadband | 66.8% |
| Away-home heat differential | 0.9°F |

Moderate behavior across all dimensions. Largest segment — represents the "typical" Ecobee user with reasonable schedule compliance and moderate comfort flexibility.

---

## Business Recommendations

**Energy Savers:**
- Energy savings dashboard with cost estimates
- Utility rebate program partnerships
- Multi-zone smart sensor upsell
- Marketing: "Save more, stay comfortable"

**Active Controllers:**
- AI-powered pattern learning to reduce manual adjustments
- Personalized automation setup consultation
- HVAC diagnostic and optimization service
- Marketing: "Take back your time — let the thermostat learn"

**Balanced Users (largest segment):**
- Engagement nudges toward energy-saving behaviors
- Feature education campaigns (Smart Away, scheduling)
- Benchmark comparisons ("You're saving X% vs similar homes")

---

## Tech Stack

- Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn
- `KMeans`, `GaussianMixture`, `DBSCAN`
- `LocalOutlierFactor`, `EmpiricalCovariance`, `KernelDensity`
- `StandardScaler`, `PCA`
- Silhouette score, Davies-Bouldin index, Calinski-Harabasz index

---

## Repository Structure

```
ecobee-user-clustering/
├── notebook.ipynb    # Full pipeline: feature engineering, clustering, profiling
├── README.md
```
