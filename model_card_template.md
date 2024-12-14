# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classifier developed to predict an individual's income being less than or equal to $50,000 or more than $50,000. Employment, education attainment, and household details are the features used to make the determination. The Random Forest Classifier was utilized.

## Intended Use
Uses cases could include demographics analysis and market research. Use of this model would align with the interests of financial institutions, social science researchers, marketing specialists, and any other party where salary prediction would prove to be useful.

## Training Data
A supervised machine learning model was trained with labeled data (collected by the US Census Bureau). Included features are mainly categorical with some being continuous. The target salary variable was converted to binary values. This model was trained with 80% of the total data available.

## Evaluation Data
Testing was performed with the remaining 20% of the dataset. Precision, recall, and the f1-score were used to determine suitability.

## Metrics
Precision: 0.7419 A score >0.70 indicates high accuracy.
Recall: 0.6384 A score of 0.60-0.70 indicates decent recall.
F1-Score: 0.6863 An f1-score between 0.5 and 0.8 is considered average.

## Ethical Considerations
This model used a limited number of features. Many factors of differing significance contribute to an individual's income and should be considered along with results obtained from this model.

## Caveats and Recommendations
Predictions for underrepresented demographic groups may not be accurate. Due to the sensitivity of the training population, this model is not recommended for use outside of the US.
