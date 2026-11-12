# 🚖 2024 Uber NCR Ride-Sharing Business Intelligence Dashboard

An interactive, production-ready Business Intelligence web application built with **Python**, **Streamlit**, **Pandas**, **SciPy**, and **Plotly** to analyze the **2024 Uber NCR Ride-Sharing Dataset** (`ncr_ride_bookings.csv`).

Designed with a dark-mode aesthetic inspired by the Uber mobile experience (`Dasboard.jpg`) and aligned with the companion Power BI report (`Uber.pbix`), this dashboard provides real-time filtering, KPI tracking, cancellation diagnostics, fleet performance comparisons, and statistical correlation modeling.

## 📌 Project Scope & Architecture

### Core Deliverables
| File | Description |
| :--- | :--- |
| `app.py` | Main Streamlit dashboard application featuring wide layout configuration, custom dark-mode CSS, global sidebar filters, dynamic KPI cards, and modular Plotly visualization functions. |
| `data_processing.py` | Modular Pandas & SciPy pipeline handling CSV path resolution, ID quote stripping, `Date` and `Time` datetime parsing, null handling across cancellation/payment columns, vehicle category grouping (`eBike/Bike`), and OLS/Pearson statistical calculations. |
| `Requirement.txt` | Pinned dependency specification (`streamlit`, `pandas`, `numpy`, `scipy`, `plotly`, `statsmodels`) preventing `ModuleNotFoundError` during local or cloud deployment. |
| `ReadMe.md` | Complete project documentation covering baseline metric verification, local VS Code setup, GitHub version control, and Streamlit Community Cloud deployment. |

## 📊 Dataset Schema & Baseline Metric Verification

The dataset `ncr_ride_bookings.csv` spans **January 1, 2024 to December 30, 2024** across 21 operational columns (`Date`, `Time`, `Booking ID`, `Booking Status`, `Customer ID`, `Vehicle Type`, `Pickup Location`, `Drop Location`, `Avg VTAT`, `Avg CTAT`, `Cancelled Rides by Customer`, `Reason for cancelling by Customer`, `Cancelled Rides by Driver`, `Driver Cancellation Reason`, `Incomplete Rides`, `Incomplete Rides Reason`, `Booking Value`, `Ride Distance`, `Driver Ratings`, `Customer Rating`, and `Payment Method`).

### Reconciliation of Baseline Metrics (`Uber.pbix` vs. Full CSV)
Inspection of the raw CSV alongside the Power BI report (`Uber.pbix`) reveals two complementary denominator conventions that explain the baseline targets:

1. **Booking Count (`148,770` vs. `150,000`)**:
   - The CSV contains **150,000 total rows** representing **148,767 unique `Booking ID`s** (which Power BI KPI cards format as **`148.77K`**, or ~148,770 bookings).
2. **Success & Cancellation Rates (`65.96%`, `25.00%`, `19.15%`, `7.45%`)**:
   - **Active-Status Basis (Excluding the 9,000 `Incomplete` rides = 141,000 rows, matching the Power BI *Overall* Booking Status visual filter)**:
     - **Success Rate (`Completed`)**: `93,000 / 141,000` = **65.96%** (`65.962%` on unique Booking IDs)
     - **Driver Cancellation Rate (`Cancelled by Driver`)**: `27,000 / 141,000` = **19.15%**
     - **Customer Cancellation Rate (`Cancelled by Customer`)**: `10,500 / 141,000` = **7.45%**
     - **No Driver Found Rate (`No Driver Found`)**: `10,500 / 141,000` = **7.45%**
   - **Full Dataset Basis (All 150,000 rows)**:
     - **Total Cancellation Rate (`Customer + Driver`)**: `(10,500 + 27,000) / 150,000` = `37,500 / 150,000` = **25.00%** (`18.00%` Driver + `7.00%` Customer)
     - **Overall Completion Rate**: `93,000 / 150,000` = **62.00%**
     - **No Driver Found**: `10,500 / 150,000` = **7.00%**
     - **Incomplete Rides**: `9,000 / 150,000` = **6.00%**
   - *Note*: In the prompt's baseline summary (`19.15% Customer, 7.45% Driver`), the labels for Customer and Driver cancellation rates are transposed relative to the underlying CSV and Power BI model, where **Driver cancellations represent 19.15%** (`27,000` rides) and **Customer cancellations represent 7.45%** (`10,500` rides). The dashboard sidebar includes a **Rate Calculation Basis** toggle so analysts can switch between both conventions on the fly.

### Key Financial, Fleet & Statistical Findings
| Metric | Full Dataset Value (All Billed Rides) | Completed Rides Only (`93,000` Rides) |
| :--- | :--- | :--- |
| **Total Revenue (`Booking Value`)** | **₹51,846,183** (₹51.85M across 102,000 billed rows) | **₹47,260,574** (₹47.26M) |
| **Total Ride Distance** | **2,512,975.19 km** | **2,418,045.84 km** |
| **Average Customer Rating** | **4.40 ★** (`4.4046`) | **4.40 ★** (`4.4046`) |
| **Average Driver Rating** | **4.23 ★** (`4.2310`) | **4.23 ★** (`4.2310`) |
| **Pearson Correlation (`r`, Distance vs. Value)** | **r = 0.0052** (*p* = 0.0984, *n* = 102,000) | **r = 0.0057** (*p* = 0.0839, *n* = 93,000) |

## 💻 Local Setup Instructions (VS Code & Virtual Environment)

### 1. Prerequisites
- **Python 3.9 – 3.12** installed on your system (`python --version`).
- **Visual Studio Code (VS Code)** with the official **Python** extension installed.
- **Git** installed (`git --version`).

### 2. Project Directory Setup
1. Create a new project folder and place the following files inside it:
   ```text
   uber-ncr-dashboard/
   ├── app.py
   ├── data_processing.py
   ├── Requirement.txt
   ├── ReadMe.md
   └── ncr_ride_bookings.csv
   ```
2. Open the folder in **VS Code** (`File` -> `Open Folder...`) and open the integrated terminal (`Ctrl + ~` on Windows/Linux or `Cmd + J` on macOS).

### 3. Create & Activate a Virtual Environment
Run the following commands in the VS Code terminal:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (Bash/Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
Install all required libraries from `Requirement.txt`:
```bash
pip install --upgrade pip
pip install -r Requirement.txt
```

### 5. Launch the Dashboard Locally
Run the Streamlit server:
```bash
streamlit run app.py
```
The dashboard will automatically open in your default browser at `http://localhost:8501`.

## 🚀 GitHub Integration & Streamlit Community Cloud Deployment

### Step 1: Prepare Repository for Streamlit Cloud
Streamlit Community Cloud automatically looks for `requirements.txt` (lowercase) or allows you to specify a custom dependency file in advanced settings. To ensure zero-friction auto-detection on Streamlit Cloud, copy `Requirement.txt` to `requirements.txt` as well:
```bash
cp Requirement.txt requirements.txt
```

### Step 2: Initialize Git & Push to GitHub
1. Create a new repository on [GitHub](https://github.com/new) (e.g., `uber-ncr-2024-dashboard`).
2. In your VS Code terminal, initialize Git, commit your files, and push to GitHub:
   ```bash
   git init
   git add app.py data_processing.py Requirement.txt requirements.txt ReadMe.md ncr_ride_bookings.csv
   git commit -m "Initial commit: 2024 Uber NCR Streamlit BI Dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/uber-ncr-2024-dashboard.git
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Configure the deployment settings:
   - **Repository**: `<your-username>/uber-ncr-2024-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**. Streamlit Cloud will provision the environment, install the packages listed in `requirements.txt` / `Requirement.txt`, and publish your live interactive dashboard within 1–2 minutes.
