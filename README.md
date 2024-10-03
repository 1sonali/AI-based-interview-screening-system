# %%
!pip install transformers torch pandas scikit-learn numpy matplotlib spacy
!python -m spacy download en_core_web_sm

# %%
import numpy as np
import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
import re

class EnhancedAIInterviewScreener:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.nlp = spacy.load("en_core_web_sm")
        
        self.technical_questions = {
            "machine_learning": [
                {
                    "question": "Explain the difference between supervised and unsupervised learning, providing examples of each.",
                    "keywords": ["labeled data", "unlabeled data", "classification", "clustering", "regression"],
                    "code_challenge": None
                },
                {
                    "question": "What is gradient descent and how does it work?",
                    "keywords": ["optimization", "learning rate", "cost function", "local minima", "iteration"],
                    "code_challenge": None
                }
            ],
            "deep_learning": [
                {
                    "question": "Explain the concept of backpropagation in neural networks.",
                    "keywords": ["chain rule", "weights", "gradients", "loss function", "optimization"],
                    "code_challenge": None
                },
                {
                    "question": "Implement a simple neural network using PyTorch for binary classification.",
                    "keywords": ["pytorch", "layers", "activation function", "forward pass", "backward pass"],
                    "code_challenge": """
def create_model():
    # Implement a simple neural network for binary classification
    pass

def train_model(model, X_train, y_train):
    # Implement training loop
    pass
"""
                }
            ],
            "data_preprocessing": [
                {
                    "question": "How do you handle missing data in a dataset?",
                    "keywords": ["imputation", "mean", "median", "deletion", "interpolation"],
                    "code_challenge": None
                },
                {
                    "question": "Implement a function to normalize numerical features in a dataset.",
                    "keywords": ["scaling", "standardization", "min-max", "z-score", "normalization"],
                    "code_challenge": """
def normalize_features(data):
    # Implement normalization function
    pass
"""
                }
            ]
        }
        
        self.behavioral_questions = [
            "Describe a challenging machine learning project you worked on and how you overcame obstacles.",
            "How do you stay updated with the latest developments in AI and machine learning?",
            "Explain how you would collaborate with non-technical stakeholders on an AI project."
        ]
        
        self.evaluation_criteria = {
            "technical_expertise": {
                "weight": 0.4,
                "subcriteria": {
                    "concept_understanding": 0.5,
                    "implementation_skills": 0.3,
                    "problem_solving": 0.2
                }
            },
            "communication": {
                "weight": 0.3,
                "subcriteria": {
                    "clarity": 0.4,
                    "technical_articulation": 0.3,
                    "engagement": 0.3
                }
            },
            "experience": {
                "weight": 0.3,
                "subcriteria": {
                    "project_complexity": 0.4,
                    "role_relevance": 0.3,
                    "impact": 0.3
                }
            }
        }

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        return cosine_similarity(embedding1, embedding2)[0][0]

    def analyze_code_solution(self, code: str, expected_keywords: List[str]) -> Dict[str, float]:
        code_lower = code.lower()
        keyword_score = sum(keyword.lower() in code_lower for keyword in expected_keywords) / len(expected_keywords)
        
        # Basic code quality checks
        code_quality_score = 0.0
        if re.search(r'def\s+\w+\s*\([^)]*\):', code):  # Has function definition
            code_quality_score += 0.2
        if re.search(r'#.*$', code, re.MULTILINE):  # Has comments
            code_quality_score += 0.2
        if re.search(r'try:.*except.*:', code, re.DOTALL):  # Has error handling
            code_quality_score += 0.2
        if re.search(r'if\s+.*:', code):  # Has conditional statements
            code_quality_score += 0.2
        if not re.search(r'while\s+True:', code):  # Avoids infinite loops
            code_quality_score += 0.2
        
        return {
            "keyword_score": keyword_score,
            "code_quality_score": code_quality_score
        }

    def evaluate_answer(self, question: Dict, answer: str) -> Dict[str, float]:
        # Initialize NLP analysis
        doc = self.nlp(answer)
        
        # Technical accuracy assessment
        keyword_presence = sum(keyword.lower() in answer.lower() for keyword in question['keywords']) / len(question['keywords'])
        
        # Semantic understanding
        semantic_score = self.calculate_semantic_similarity(question['question'], answer)
        
        # Communication clarity
        sentence_count = len(list(doc.sents))
        avg_sentence_length = len(doc) / sentence_count if sentence_count > 0 else 0
        communication_score = 1.0 if 10 <= avg_sentence_length <= 20 else 0.5
        
        technical_terms = sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() in question['keywords'])
        technical_articulation = min(technical_terms / len(question['keywords']), 1.0)
        
        scores = {
            "keyword_presence": keyword_presence,
            "semantic_understanding": float(semantic_score),
            "communication_clarity": communication_score,
            "technical_articulation": technical_articulation
        }
        
        if question.get('code_challenge'):
            code_scores = self.analyze_code_solution(answer, question['keywords'])
            scores.update(code_scores)
        
        return scores

    def calculate_final_score(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        final_scores = {}
        
        for criteria, details in self.evaluation_criteria.items():
            criteria_score = 0
            for subcriteria, weight in details['subcriteria'].items():
                subcriteria_scores = [
                    score.get(subcriteria, 0) for score in all_scores if subcriteria in score
                ]
                if subcriteria_scores:
                    criteria_score += weight * np.mean(subcriteria_scores)
            
            final_scores[criteria] = criteria_score * details['weight']
        
        final_scores['overall_score'] = sum(final_scores.values())
        return final_scores

class InterviewSimulator:
    def __init__(self, screener: EnhancedAIInterviewScreener):
        self.screener = screener
        
    def generate_candidate_response(self, question: Dict, quality: str) -> str:
        quality_factors = {
            "excellent": 0.9,
            "good": 0.7,
            "average": 0.5,
            "poor": 0.3
        }
        
        factor = quality_factors[quality]
        keywords = question['keywords']
        
        if question.get('code_challenge'):
            return self.generate_code_response(question['code_challenge'], keywords, factor)
        else:
            return self.generate_text_response(question['question'], keywords, factor)
    
    def generate_text_response(self, question: str, keywords: List[str], quality_factor: float) -> str:
        base_response = f"Regarding {question.split()[0].lower()} "
        selected_keywords = np.random.choice(
            keywords, 
            size=int(len(keywords) * quality_factor),
            replace=False
        )
        
        for keyword in selected_keywords:
            base_response += f"{keyword}, "
        
        return base_response.rstrip(", ") + "."
    
    def generate_code_response(self, template: str, keywords: List[str], quality_factor: float) -> str:
        if quality_factor < 0.5:
            return template  # Return unchanged template for poor responses
        
        # Simple code generation based on quality
        code_lines = []
        if quality_factor >= 0.7:  # Good or excellent
            code_lines.append("import numpy as np")
            code_lines.append("import torch")
            code_lines.append("import torch.nn as nn")
            
        code_lines.extend(template.split('\n'))
        
        if quality_factor >= 0.9:  # Excellent
            code_lines.append("    # Add error handling")
            code_lines.append("    try:")
            code_lines.append("        # Implementation")
            code_lines.append("    except Exception as e:")
            code_lines.append("        print(f'Error: {e}')")
        
        return "\n".join(code_lines)

    def simulate_interviews(self, num_candidates: int) -> pd.DataFrame:
        qualities = ["excellent", "good", "average", "poor"]
        probabilities = [0.2, 0.3, 0.3, 0.2]
        
        all_results = []
        
        for i in range(num_candidates):
            candidate_quality = np.random.choice(qualities, p=probabilities)
            candidate_responses = []
            
            # Technical questions
            for category, questions in self.screener.technical_questions.items():
                for question in questions:
                    response = self.generate_candidate_response(question, candidate_quality)
                    scores = self.screener.evaluate_answer(question, response)
                    candidate_responses.append(scores)
            
            # Calculate final score
            final_scores = self.screener.calculate_final_score(candidate_responses)
            final_scores['candidate_id'] = f"Candidate_{i+1}"
            final_scores['true_quality'] = candidate_quality
            
            all_results.append(final_scores)
        
        return pd.DataFrame(all_results)

def plot_results(results_df: pd.DataFrame):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Score distribution
    plt.subplot(2, 1, 1)
    score_columns = [col for col in results_df.columns if col.endswith('_score') or col in ['technical_expertise', 'communication', 'experience']]
    results_df[score_columns].boxplot()
    plt.title('Distribution of Candidate Scores')
    plt.xticks(rotation=45)
    
    # Plot 2: Quality vs Overall Score
    plt.subplot(2, 1, 2)
    sns.boxplot(x='true_quality', y='overall_score', data=results_df)
    plt.title('Overall Score by Candidate Quality')
    
    plt.tight_layout()
    plt.show()

# Run simulation
screener = EnhancedAIInterviewScreener()
simulator = InterviewSimulator(screener)
results = simulator.simulate_interviews(20)

# Display results
print("Interview Results:")
print(results.sort_values('overall_score', ascending=False))

# Plot results
plot_results(results)

# Example of evaluating a single candidate manually
def evaluate_single_candidate():
    print("\nEvaluating single candidate example:")
    
    # Example technical question response
    technical_question = screener.technical_questions['machine_learning'][0]
    technical_response = """
    Supervised learning uses labeled data where the model learns to predict outputs based on input features. 
    For example, in image classification, the model learns from images labeled with categories. 
    Unsupervised learning works with unlabeled data to find patterns or structures, like clustering customers based on behavior.
    """
    
    technical_scores = screener.evaluate_answer(technical_question, technical_response)
    print("\nTechnical Question Evaluation:")
    for criterion, score in technical_scores.items():
        print(f"{criterion}: {score:.2f}")
    
    # Example code challenge response
    code_question = screener.technical_questions['deep_learning'][1]
    code_response = """
    import torch
    import torch.nn as nn

    def create_model():
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        return model

    def train_model(model, X_train, y_train):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        try:
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
        except Exception as e:
            print(f'Error during training: {e}')
    """
    
    code_scores = screener.evaluate_answer(code_question, code_response)
    print("\nCode Challenge Evaluation:")
    for criterion, score in code_scores.items():
        print(f"{criterion}: {score:.2f}")

evaluate_single_candidate()
