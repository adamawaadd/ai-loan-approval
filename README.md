# AI Loan Approval Project:

This project is a term-long learning experience regarding artificial intelligence, machine learning fundamentals, and model development in python. I chose a loan-approval dataset because I’m passionate about finance (and currently working toward a finance minor), making this a perfect beginner application of my interests.

Throughout the project, I worked with a variety of Python data and ML tools at a basic level, including pandas, NumPy, Matplotlib, SciKit-Learn, YData Profiling, and Joblib. I used them to explore the data, engineer features, scale data, one-hot encode, build my model, and visualize results.

Although this repository’s first commit was about halfway through the project, a large amount of experimentation happened earlier. I tested multiple models and ultimately selected a Decision Tree Classifier because it provides a human-interpretable reasoning path—an important feature for something as sensitive and consequential as loan approval. I tracked experimentation, feature engineering notes, and model performance in progress.txt.

The "loan_approval_preliminary_data.html" file is downloadable and basically just shows a visualization of the distribution of each column of data (such as the distribution of the sampled group's age, income, employment experience, etc.)

## Installation + Setup:
1.) Clone the repository
```
git clone https://github.com/adamawaadd/ai-loan-approval.git
cd ai-loan-approval
```

2.) Make sure your python version is 3.9 or higher

3.) Install pandas, numpy, matplotlib, scikit-learn, and joblib
```
pip install numpy pandas matplotlib scikit-learn joblib
```

4.) Run the program!
```
python loan_approval.py
```

## To-Do:
1.) Make sure my test split is no more than ~5% worse than my training split

2.) Add the HTML file to .gitignore
