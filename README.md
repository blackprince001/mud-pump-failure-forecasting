# **Executive Summary**

Our objective was to build a machine learning model to predict failures (identified as 'STOP' events) for two mud pumps, A and B. Our process was a multi-phased journey that began with Exploratory Data Analysis (EDA), moved to complex deep learning with LSTMs, and finally pivoted to powerful tree-based models like Random Forest and XGBoost.

Along the way, we encountered and solved several critical, real-world machine learning challenges. Our initial models produced either poor, unusable results or, more dangerously, **"too good to be true" results with over 98% accuracy**. These perfect scores were not a sign of success but a red flag for critical flaws in our methodology.

Through careful debugging, we identified two severe forms of **data leakage**:

1. **Trivial Prediction:** The model was not predicting failures but simply identifying pumps that were *already stopped* by looking at trivial features like `Speed = 0`.
2. **Seeing the Future:** The data was being split randomly instead of chronologically, allowing the model to train on future data to "predict" the past.

By correcting these fundamental errors, we developed a final, robust, and genuinely predictive methodology. This final approach focuses on the correct question—**"Is the pump showing signs that it is *about to* fail?"**—and uses a time-aware validation strategy that simulates a real-world deployment.

## **Phase 1: Foundation - Data Exploration and Feature Engineering**

This initial phase was about understanding and preparing the data for any type of model.

* **Data Cleaning:** We loaded the data for both pumps and immediately faced our first challenge: non-numeric values like `'Bad'` hidden in supposedly numeric sensor columns.
  * **Solution:** We implemented a robust cleaning function to systematically convert all feature columns to a numeric format, replacing invalid strings with `NaN` and then filling them with the last known valid reading (`ffill`). This prevented the models from crashing.

* **Exploratory Data Analysis (EDA):** To understand the pumps' behavior, we:
  * **Standardized Column Names:** Aligned names like `MUD PUMP A SUCTION PRESS` to a common `Suction Pressure` to allow for direct comparison.
  * **Visualized Feature Histories:** Plotted the time-series data for every sensor to visually inspect trends, seasonality, and anomalies.
  * **Compared Distributions:** Used histograms and density plots to see if the operational ranges (e.g., average current, pressure) differed between Pump A and Pump B.

* **Feature Engineering:** We recognized that a single data point in time has limited context.
  * **Solution:** We created **rolling statistics** (mean, standard deviation) over a 24-hour window. This provided the models with crucial context about recent trends and stability, transforming the data from a simple snapshot to a richer, more informative feature set.

## **Phase 2: The Deep Learning Approach with LSTMs**

Our first modeling attempt used LSTMs, which are well-suited for sequential data.

* **Initial Goal (Flawed):** We started with a highly ambitious goal: use 168 hours (7 days) of data to predict the exact status for the *next* 168 hours.
  * **Problem:** The results were poor. For Pump A, the model had high recall (it found most failures) but terrible precision (it "cried wolf" constantly, generating too many false alarms). The task was simply too complex and speculative.

* **Iteration and Improvement:** We refined our strategy to make the problem more manageable.
  * **Solution:** We reframed the goal to a more realistic **72-hour input -> 72-hour prediction window**. We also upgraded the model to a **Bidirectional LSTM (BiLSTM)** to better capture patterns and added a **Learning Rate Scheduler** to improve training. This significantly improved the results, making them more balanced and useful.

## **Phase 3: The Pivot to Tree-Based Models & Uncovering Critical Flaws**

Recognizing the power of traditional machine learning, we pivoted to **Random Forest** and **XGBoost**. This is where we uncovered the most critical errors.

* **Feature Preparation:** We "flattened" the 72-hour input sequences into a single feature vector for each training sample, which is the standard method for tree-based models.

* **The "Too Good to Be True" Problem:** The initial results were staggering, with **98-99% accuracy, precision, and recall**. This was an immediate red flag.

* **Uncovering Data Leakage #1: Trivial Prediction**
  * **The Flaw:** We realized the models were not learning the subtle patterns *before* a failure. Instead, they learned a simple cheat: if the pump's `Speed` or `Current` in the last hour of the input data was 0, the pump was already stopped. The model was simply reporting on a state that had already occurred.
  * **The Lesson:** The target variable was flawed. We were asking, "Is the pump stopped?" when we should have been asking, "Is the pump *about to* stop?".

* **Uncovering Data Leakage #2: Incorrect Data Splitting**
  * **The Flaw:** We were using a standard `train_test_split`, which **randomly shuffles** data. For time-series, this is catastrophic. It meant our model could be trained on data from November and tested on its ability to "predict" an event in August. It was learning from the future.
  * **The Lesson:** Time-series data must **always be split chronologically** (e.g., first 80% for training, last 20% for testing) to simulate a real-world scenario where the future is unknown.

## **Phase 4: The Final, Robust, and Correct Methodology**

Armed with the lessons from our previous attempts, we built a final, leak-proof methodology that represents the correct way to approach this problem.

1. **Redefine the Prediction Target:**
    * We stopped trying to predict the 'STOP' state itself.
    * We created a **"pre-failure warning window"**—a new target that is flagged as `1` for the 6 hours *leading up to* a failure. All other 'RUN' periods are `0`.
    * Crucially, we **removed all data points where the pump was already stopped** from the dataset, completely eliminating the "cheat code."

2. **Implement a Chronological Split:**
    * We replaced the random split with a simple chronological split, ensuring the model trains only on past data to predict the future.

This final methodology forces the model to solve the correct and valuable business problem: **distinguishing between normal operation and the subtle anomalous behavior that precedes a failure, using only historical data.** The results from this approach will be more modest but will reflect the model's true, generalizable predictive power.
