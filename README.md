# Automatic-Essay-Grading-Using-NLP-Features
"This project applies Deep Learning for machine fault detection using acoustic data. It classifies Arcing, Corona, Looseness, and Tracking faults via Mel Spectrogram images and a CNN model. The system ensures accurate fault identification with an 80/20 data split and visualizes performance using a confusion matrix."


 
Automatic Essay Grading Using NLP Features
NLP Semester Project


 
Umer Abid
2021-SE-03
 
Sallah Udin
2021-SE-18
 
Rafaqat Ali
2021-SE-30 



Abstract
The Automatic Grading Essay System leverages Natural Language Processing (NLP) techniques for scoring essays. Developed pipeline processes include data loading, text cleaning, feature extraction through TF-IDF, and model training with the XGBoost algorithm. The target goal is to accurately classify essays into specific score ranges. Consequently, the model performs impressively with roughly 90% accuracy and an F1 score of 90%. The project aims to provide faster essay evaluation and reduction of manual labor in the	educational	field.



          				  Introduction

Automatic scoring of essays is integral to today's educational systems as it saves time and provides consistency in scoring as opposed to manual grading. Grading English essays has struggled to sustain cadence as devised whereas coming up with a grading strategy for diverse written forms and styles of English has its own share of challenges. This project uses a combination of NLP and machine learning algorithms such as XGboost to meet these challenges	and	achieve	balanced	results. 

As the Amharic language modeling project was based on developing a social media corpus, subsequent data modeling steps followed built the pipeline off consisting of data cleaning, tokenization, feature extraction and trained classifiers. A well-defined projection design improves model performance and results, while evaluation metrics such as accuracy and an F1 score demonstrate system efficiency.
						Related Work
Earlier methods for automated essay scoring relied on algorithms of classical machine learning like Naïve Bayes, Decision Trees, and Support Vector Machines. Unfortunately, these approaches did not perform well with high level contextual information. CNNs and RNNs performed better with respect to text comprehension as they are deep learning models. Current models of transformers like BERT and XLM-Roberta have been proved to outperform other models in NLP activities. However, these models offer a more accurate approach and this project proposes a more time-efficient strategy by incorporating TF-IDF weighting with XGBoost for effective grading.

●	1. Proposed Methodology 

●	1.1. Data Acquisition and Preprocessing 
 
 
     Loading the dataset. 

In this step, the data that contains the essays that need to be graded is loaded through Pandas. The essays are text documents available in different lengths with the corresponding grades. 

Splitting the dataset. 
The model is trained using seventy percent of the data, and fifteen percent is assigned for model validation and test set for unbiased examination of the model performance. 

Measuring cleanliness of data.
 The process of data cleaning entails no punctuation, no special characters, no whitespace, and non-meaningful symbols. This changes the case of the words and lemmatization is done stemming the text to base words. 

Sentiment mapping.
 In the model, accounts are grouped to relevant primary sentiments to negative, neutral, or supportive. Models require such class assignments during the training process to ensure improved learning results.
Dataset Balancing

In order to prevent class imbalance, a sentiment-balanced subset is sampled from each sentiment category. The balanced data is visualized through bar plots and then transformed into a Hugging Face Dataset. 

● Dataset Splitting 

To provide a fair evaluation for the model during training, the dataset is distributed into training (80%), validation (10%), and test (10%) portions. 

1.2.	Tokenization and Feature Engineering 

TF-IDF Vectorization: Texts are transformed into numbers by the TF-IDF approach with a cap of 10,000 features and an n-gram range of (1, 3) to guarantee substantial text representation. 

1.3.	Tokenization and Feature Extraction 

● Tokenizer Setup 

The Urdu text is processed with the "xlm-roberta-base" tokenizer. A custom tokenization function implements truncation and padding (max_length=10000) to equalize the input size across the entire	dataset. 
●	Embedding Representation

The pretrained transformer, which naturally captures syntactic and semantic information in Urdu, is used to turn tokens into dense embeddings.
1.2.	Model Architecture and Training
Model Configuration: 
The XGBoost Classifier is selected due to its efficiency and strong performance on text data.
Training Parameters:
Learning Rate: 0.1
Number of Trees: 500
Maximum Depth: 15
Evaluation:
High performance is obtained from post-training evaluation on the test set (about 98% accuracy and F1-score). Insights into model performance across sentiment categories are provided by detailed classification reports and confusion matrix representations.

 

1.3.	Model Deployment on Hugging Face Spaces

Evaluation And Results:
The trained model achieves ~90% accuracy and an F1-score of 90% on the test set. The confusion matrix illustrates the model's performance across score ranges.
●	Limitations:

Dataset Size: Although balanced, the dataset might not fully capture diverse essay structures, limiting model generalization.
Complex Language Patterns: Urdu text exhibits complex sentence structures that may reduce model efficiency. Further linguistic improvements could mitigate this issue.
Model Complexity: While XGBoost performs well, experimenting with ensemble models or transformers may further improve results.
 
●	Web Interface Interaction
The Gradio interface includes textboxes for input and output, a descriptive title, and examples demonstrating usage. This deployment enables real-time sentiment prediction and broad accessibility via the Hugging Face Spaces platform.
2.	Limitations

While the project shows promising results, while several limitations remain:

●	Dataset Limitations
The dataset, though balanced for the experiment, may not fully capture the diverse expressions of sentiment in Urdu. Broader datasets could help improve model robustness.
●	Preprocessing Challenges
However useful, custom text cleaning and tokenization for Urdu might not take into account all language complexities, which could compromise accuracy in edge situations.
●	Model Complexity
The current approach uses a fine-tuned transformer with a straightforward classifier. More sophisticated architectures or ensemble methods might further enhance performance, particularly in noisy real-world environments.
 
3.	Conclusions
This project successfully demonstrates a complete pipeline for automatic essay grading using NLP techniques. By combining robust text cleaning, TF-IDF features, and XGBoost, the system achieves impressive results. Future improvements could explore transformer models like BERT or incorporate semantic similarity scoring for improved understanding.

4.	References

1.	Pang, B., & Lee, L. (2008). Opinion Mining and Sentiment Analysis. Foundations and Trends in Information Retrieval.
2.	Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
3.	Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NIPS.
4.	Wolf, T., et al. (2020). Transformers: State-of-the-art Natural Language Processing. EMNLP.
5.	Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.
1.	
