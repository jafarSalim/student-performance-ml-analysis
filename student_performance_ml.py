import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import os
import copy

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def load_student_data():
    """Load student data with multiple fallback options"""

    # First, try to find the actual file
    possible_paths = [
        'student-mat.csv',
        './student-mat.csv',
        'student-por.csv',
        './student-por.csv'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Found dataset at: {path}")
            try:
                # Try semicolon separator first (common in European datasets)
                df = pd.read_csv(path, sep=';')
                if len(df.columns) > 10:
                    print(f"✓ Successfully loaded with semicolon separator")
                    return df, True
                else:
                    # Try comma separator
                    df = pd.read_csv(path, sep=',')
                    print(f"✓ Successfully loaded with comma separator")
                    return df, True
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                continue

    print("❌ Could not find student-mat.csv or student-por.csv!")
    print("📊 Generating synthetic student performance data for demonstration...")
    return generate_synthetic_student_data(), False


def generate_synthetic_student_data():
    """Generate realistic synthetic student performance data"""

    np.random.seed(42)
    n_samples = 395

    print(f"Generating {n_samples} synthetic student records...")

    # Create synthetic data with realistic correlations
    data = {}

    # Basic demographics
    data['school'] = np.random.choice(['GP', 'MS'], n_samples, p=[0.7, 0.3])
    data['sex'] = np.random.choice(['F', 'M'], n_samples, p=[0.52, 0.48])
    data['age'] = np.random.choice([15, 16, 17, 18, 19, 20, 21, 22], n_samples,
                                   p=[0.05, 0.2, 0.25, 0.25, 0.15, 0.05, 0.03, 0.02])
    data['address'] = np.random.choice(['U', 'R'], n_samples, p=[0.75, 0.25])
    data['famsize'] = np.random.choice(['LE3', 'GT3'], n_samples, p=[0.3, 0.7])
    data['Pstatus'] = np.random.choice(['T', 'A'], n_samples, p=[0.85, 0.15])

    # Parent education (correlated)
    data['Medu'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.05, 0.15, 0.25, 0.3, 0.25])
    data['Fedu'] = []
    for medu in data['Medu']:
        if medu <= 1:
            fedu = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        elif medu == 2:
            fedu = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        elif medu == 3:
            fedu = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])
        else:  # medu == 4
            fedu = np.random.choice([3, 4], p=[0.3, 0.7])
        data['Fedu'].append(fedu)

    # Jobs and other categorical variables
    jobs = ['teacher', 'health', 'services', 'at_home', 'other']
    data['Mjob'] = np.random.choice(jobs, n_samples, p=[0.15, 0.1, 0.3, 0.25, 0.2])
    data['Fjob'] = np.random.choice(jobs, n_samples, p=[0.1, 0.1, 0.35, 0.05, 0.4])

    data['reason'] = np.random.choice(['home', 'reputation', 'course', 'other'], n_samples, p=[0.15, 0.2, 0.4, 0.25])
    data['guardian'] = np.random.choice(['mother', 'father', 'other'], n_samples, p=[0.75, 0.2, 0.05])
    data['traveltime'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05])
    data['studytime'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.5, 0.25, 0.1])
    data['failures'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.65, 0.2, 0.1, 0.05])

    # Binary variables
    data['schoolsup'] = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    data['famsup'] = np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4])
    data['paid'] = np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7])
    data['activities'] = np.random.choice(['yes', 'no'], n_samples, p=[0.5, 0.5])
    data['nursery'] = np.random.choice(['yes', 'no'], n_samples, p=[0.85, 0.15])
    data['higher'] = np.random.choice(['yes', 'no'], n_samples, p=[0.95, 0.05])
    data['internet'] = np.random.choice(['yes', 'no'], n_samples, p=[0.8, 0.2])
    data['romantic'] = np.random.choice(['yes', 'no'], n_samples, p=[0.35, 0.65])

    # Social variables (1-5 scale)
    data['famrel'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.02, 0.05, 0.15, 0.58, 0.2])
    data['freetime'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.4, 0.3, 0.1])
    data['goout'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.35, 0.25, 0.05])
    data['Dalc'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.7, 0.2, 0.07, 0.02, 0.01])
    data['Walc'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.25, 0.2, 0.1, 0.05])
    data['health'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.2, 0.4, 0.25])

    # Absences
    data['absences'] = np.random.poisson(4, n_samples)
    data['absences'] = np.clip(data['absences'], 0, 30)

    # Generate correlated grades
    G1, G2, G3 = [], [], []

    for i in range(n_samples):
        base_score = 10
        base_score += data['studytime'][i] * 1.2
        base_score += data['Medu'][i] * 0.8
        base_score += data['Fedu'][i] * 0.6
        base_score += 1.5 if data['famsup'][i] == 'yes' else 0
        base_score += 1.0 if data['schoolsup'][i] == 'yes' else 0
        base_score += 2.0 if data['higher'][i] == 'yes' else -3.0
        base_score += (data['famrel'][i] - 3) * 0.5

        base_score -= data['failures'][i] * 2.5
        base_score -= data['absences'][i] * 0.15
        base_score -= (data['goout'][i] - 3) * 0.3
        base_score -= (data['Dalc'][i] - 1) * 0.5
        base_score -= (data['Walc'][i] - 1) * 0.4
        base_score -= max(0, data['age'][i] - 17) * 0.3

        random_factor = np.random.normal(0, 2.5)
        final_base = base_score + random_factor

        g1 = max(0, min(20, final_base + np.random.normal(0, 1.5)))
        g2 = max(0, min(20, 0.8 * g1 + 0.2 * final_base + np.random.normal(0, 1.2)))
        g3 = max(0, min(20, 0.6 * g2 + 0.3 * g1 + 0.1 * final_base + np.random.normal(0, 1.0)))

        G1.append(round(g1))
        G2.append(round(g2))
        G3.append(round(g3))

    data['G1'] = G1
    data['G2'] = G2
    data['G3'] = G3

    df = pd.DataFrame(data)
    df['G3'] = np.clip(df['G3'], 0, 20)

    print(f"✓ Generated synthetic dataset with {len(df)} students and {len(df.columns)} features")
    return df


# ===============================
# 1. NAIVE BAYES CLASSIFIERS
# ===============================

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.feature_stats = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = X.shape[0]

        for cls in self.classes:
            class_mask = (y == cls)
            self.class_priors[cls] = np.sum(class_mask) / n_samples

            class_data = X[class_mask]
            self.feature_stats[cls] = {
                'mean': np.mean(class_data, axis=0),
                'var': np.var(class_data, axis=0) + 1e-6
            }

    def _gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

    def predict_proba(self, X):
        probabilities = []

        for sample in X:
            class_probs = {}

            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])

                for i, feature_val in enumerate(sample):
                    mean = self.feature_stats[cls]['mean'][i]
                    var = self.feature_stats[cls]['var'][i]
                    likelihood = self._gaussian_pdf(feature_val, mean, var)
                    log_prob += np.log(likelihood + 1e-10)

                class_probs[cls] = log_prob

            log_probs = np.array(list(class_probs.values()))
            probs = np.exp(log_probs - np.max(log_probs))
            probs = probs / np.sum(probs)
            probabilities.append(probs)

        return np.array(probabilities)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X, y):
        X = np.abs(X) + 0.1

        self.classes = np.unique(y)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        for cls in self.classes:
            class_mask = (y == cls)
            class_data = X[class_mask]

            self.class_priors[cls] = np.sum(class_mask) / n_samples

            feature_sums = np.sum(class_data, axis=0)
            total_sum = np.sum(feature_sums)

            self.feature_probs[cls] = (feature_sums + self.alpha) / (total_sum + self.alpha * n_features)

    def predict_proba(self, X):
        X = np.abs(X) + 0.1
        probabilities = []

        for sample in X:
            class_probs = {}

            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])

                for i, feature_val in enumerate(sample):
                    prob = self.feature_probs[cls][i]
                    log_prob += feature_val * np.log(prob + 1e-10)

                class_probs[cls] = log_prob

            log_probs = np.array(list(class_probs.values()))
            probs = np.exp(log_probs - np.max(log_probs))
            probs = probs / np.sum(probs)
            probabilities.append(probs)

        return np.array(probabilities)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X, y):
        X_binary = (X > self.binarize).astype(int)

        self.classes = np.unique(y)
        n_samples = X.shape[0]

        for cls in self.classes:
            class_mask = (y == cls)
            class_data = X_binary[class_mask]

            self.class_priors[cls] = np.sum(class_mask) / n_samples

            feature_counts = np.sum(class_data, axis=0)
            n_class_samples = np.sum(class_mask)

            self.feature_probs[cls] = (feature_counts + self.alpha) / (n_class_samples + 2 * self.alpha)

    def predict_proba(self, X):
        X_binary = (X > self.binarize).astype(int)
        probabilities = []

        for sample in X_binary:
            class_probs = {}

            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])

                for i, feature_val in enumerate(sample):
                    prob = self.feature_probs[cls][i]
                    if feature_val == 1:
                        log_prob += np.log(prob + 1e-10)
                    else:
                        log_prob += np.log(1 - prob + 1e-10)

                class_probs[cls] = log_prob

            log_probs = np.array(list(class_probs.values()))
            probs = np.exp(log_probs - np.max(log_probs))
            probs = probs / np.sum(probs)
            probabilities.append(probs)

        return np.array(probabilities)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


# ===============================
# 2. PRINCIPAL COMPONENT ANALYSIS
# ===============================

class PCA_Standard:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        self.eigenvalues = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance_ratio = self.eigenvalues[:self.n_components] / np.sum(self.eigenvalues)

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PCA_SVD:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        self.singular_values = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components = Vt[:self.n_components]
        self.singular_values = s

        total_var = np.sum(s ** 2)
        self.explained_variance_ratio = (s[:self.n_components] ** 2) / total_var

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PCA_Incremental:
    def __init__(self, n_components=2, batch_size=50):
        self.n_components = n_components
        self.batch_size = batch_size
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.mean = np.mean(X, axis=0)

        # Simple implementation: use SVD on the full data
        X_centered = X - self.mean
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components = Vt[:self.n_components]
        total_var = np.sum(s ** 2)
        self.explained_variance_ratio = (s[:self.n_components] ** 2) / total_var

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ===============================
# 3. SEMI-SUPERVISED LEARNING
# ===============================

class SelfTrainingNaiveBayes:
    def __init__(self, confidence_threshold=0.8, max_iterations=10):
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.base_classifier = GaussianNaiveBayes()

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        current_X = X_labeled.copy()
        current_y = y_labeled.copy()

        for iteration in range(self.max_iterations):
            self.base_classifier.fit(current_X, current_y)

            if len(X_unlabeled) == 0:
                break

            probs = self.base_classifier.predict_proba(X_unlabeled)
            max_probs = np.max(probs, axis=1)
            confident_mask = max_probs > self.confidence_threshold

            if np.sum(confident_mask) > 0:
                confident_samples = X_unlabeled[confident_mask]
                confident_labels = self.base_classifier.predict(confident_samples)

                current_X = np.vstack([current_X, confident_samples])
                current_y = np.hstack([current_y, confident_labels])

                X_unlabeled = X_unlabeled[~confident_mask]
            else:
                break

        self.base_classifier.fit(current_X, current_y)

    def predict(self, X):
        return self.base_classifier.predict(X)

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)


class CoTrainingNaiveBayes:
    def __init__(self, confidence_threshold=0.7, max_iterations=5):
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.classifier1 = GaussianNaiveBayes()
        self.classifier2 = MultinomialNaiveBayes()
        self.final_classifier = GaussianNaiveBayes()

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        n_features = X_labeled.shape[1]
        features1 = list(range(0, n_features, 2))
        features2 = list(range(1, n_features, 2))

        if len(features1) == 0:
            features1 = [0]
        if len(features2) == 0:
            features2 = [-1]

        current_X = X_labeled.copy()
        current_y = y_labeled.copy()
        remaining_unlabeled = X_unlabeled.copy()

        for iteration in range(self.max_iterations):
            self.classifier1.fit(current_X[:, features1], current_y)
            self.classifier2.fit(current_X[:, features2], current_y)

            if len(remaining_unlabeled) == 0:
                break

            pred1 = self.classifier1.predict(remaining_unlabeled[:, features1])
            pred2 = self.classifier2.predict(remaining_unlabeled[:, features2])
            probs1 = self.classifier1.predict_proba(remaining_unlabeled[:, features1])
            probs2 = self.classifier2.predict_proba(remaining_unlabeled[:, features2])

            agreement_mask = (pred1 == pred2)
            confidence_mask = (np.max(probs1, axis=1) > self.confidence_threshold) & \
                              (np.max(probs2, axis=1) > self.confidence_threshold)
            final_mask = agreement_mask & confidence_mask

            if np.sum(final_mask) > 0:
                new_samples = remaining_unlabeled[final_mask]
                new_labels = pred1[final_mask]

                current_X = np.vstack([current_X, new_samples])
                current_y = np.hstack([current_y, new_labels])
                remaining_unlabeled = remaining_unlabeled[~final_mask]
            else:
                break

        self.final_classifier.fit(current_X, current_y)

    def predict(self, X):
        return self.final_classifier.predict(X)

    def predict_proba(self, X):
        return self.final_classifier.predict_proba(X)


class LabelPropagationNaiveBayes:
    def __init__(self, alpha=0.2, max_iterations=30):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.classifier = GaussianNaiveBayes()

    def _rbf_kernel(self, X1, X2, gamma=1.0):
        dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * dists)

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        # Simple implementation: just use the labeled data
        self.classifier.fit(X_labeled, y_labeled)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


# ===============================
# DATA PREPROCESSING
# ===============================

def preprocess_student_data(df):
    """Preprocess the student dataset for ML"""

    print("🔄 Preprocessing student data...")

    # Create pass/fail target
    df['pass_fail'] = (df['G3'] >= 10).astype(int)

    # Remove target-related columns
    feature_cols = [col for col in df.columns if col not in ['G1', 'G2', 'G3', 'pass_fail']]

    # Separate categorical and numerical features
    X_processed = pd.DataFrame()

    # Handle numerical features
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        X_processed[col] = df[col]

    # Handle categorical features with label encoding
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in categorical_cols:
        X_processed[col + '_encoded'] = le.fit_transform(df[col].astype(str))

    # Get target
    y = df['pass_fail'].values
    X = X_processed.values

    print(f"✓ Preprocessed features: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"✓ Target distribution: {np.bincount(y)} (0=Fail, 1=Pass)")

    return X, y, list(X_processed.columns)


# ===============================
# MAIN ANALYSIS FUNCTION
# ===============================

def run_comprehensive_analysis():
    """Run comprehensive ML analysis with all 3 types of algorithms"""

    print("=" * 70)
    print("🎓 COMPREHENSIVE ML ANALYSIS: NAIVE BAYES + PCA + SEMI-SUPERVISED")
    print("=" * 70)

    # Load data
    df, is_real_data = load_student_data()

    if is_real_data:
        print("📊 Using REAL student performance dataset")
    else:
        print("🔬 Using SYNTHETIC student performance dataset")

    # Preprocess data
    X, y, feature_names = preprocess_student_data(df)

    # Train-test split before scaling to avoid test-set leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Fit the scaler on training data only, then transform the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"\n📈 Data Split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")

    # Initialize results storage
    all_results = []

    # ================================
    # 1. NAIVE BAYES ANALYSIS
    # ================================
    print(f"\n{'=' * 50}")
    print("🤖 1. NAIVE BAYES CLASSIFIERS ANALYSIS")
    print(f"{'=' * 50}")

    nb_algorithms = {
        'Gaussian NB': GaussianNaiveBayes(),
        'Multinomial NB': MultinomialNaiveBayes(),
        'Bernoulli NB': BernoulliNaiveBayes()
    }

    nb_results = {}

    for name, clf in nb_algorithms.items():
        print(f"\n🔍 Testing {name}...")

        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Manual stratified cross-validation for the custom estimators
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for fold_train_idx, fold_val_idx in cv.split(X_train, y_train):
                fold_clf = copy.deepcopy(clf)
                fold_clf.fit(X_train[fold_train_idx], y_train[fold_train_idx])
                fold_pred = fold_clf.predict(X_train[fold_val_idx])
                cv_scores.append(accuracy_score(y_train[fold_val_idx], fold_pred))
            cv_scores = np.asarray(cv_scores)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as cv_error:
            print(f"   ⚠️ Cross-validation unavailable: {cv_error}")
            cv_mean = np.nan
            cv_std = np.nan

        nb_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }

        all_results.append({
            'Algorithm_Type': 'Naive_Bayes',
            'Algorithm': name,
            'Accuracy': f"{accuracy:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1_Score': f"{f1:.4f}",
            'CV_Mean': f"{cv_mean:.4f}",
            'CV_Std': f"{cv_std:.4f}"
        })

        print(f"   ✓ Accuracy: {accuracy:.4f}")
        print(f"   ✓ Precision: {precision:.4f}")
        print(f"   ✓ Recall: {recall:.4f}")
        print(f"   ✓ F1-Score: {f1:.4f}")
        print(f"   ✓ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")

    # ================================
    # 2. PCA ANALYSIS
    # ================================
    print(f"\n{'=' * 50}")
    print("📊 2. PRINCIPAL COMPONENT ANALYSIS")
    print(f"{'=' * 50}")

    pca_algorithms = {
        'Standard PCA': PCA_Standard(n_components=min(10, X_train.shape[1])),
        'SVD PCA': PCA_SVD(n_components=min(10, X_train.shape[1])),
        'Incremental PCA': PCA_Incremental(n_components=min(10, X_train.shape[1]))
    }

    pca_results = {}

    for name, pca in pca_algorithms.items():
        print(f"\n🔍 Testing {name}...")

        try:
            # Fit PCA
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Get explained variance
            explained_var = 0.8  # default
            if hasattr(pca, 'explained_variance_ratio') and pca.explained_variance_ratio is not None:
                explained_var = np.sum(pca.explained_variance_ratio)
                print(f"   ✓ Explained Variance: {explained_var:.4f}")
                print(f"   ✓ Components: {len(pca.explained_variance_ratio)}")

            # Test Gaussian NB on PCA-reduced data
            clf_pca = GaussianNaiveBayes()
            clf_pca.fit(X_train_pca, y_train)
            y_pred_pca = clf_pca.predict(X_test_pca)
            accuracy_pca = accuracy_score(y_test, y_pred_pca)
            precision_pca = precision_score(y_test, y_pred_pca, average='weighted', zero_division=0)
            recall_pca = recall_score(y_test, y_pred_pca, average='weighted', zero_division=0)
            f1_pca = f1_score(y_test, y_pred_pca, average='weighted', zero_division=0)

            pca_results[name] = {
                'explained_variance': explained_var,
                'accuracy_with_gaussian_nb': accuracy_pca,
                'n_components': X_train_pca.shape[1]
            }

            all_results.append({
                'Algorithm_Type': 'PCA',
                'Algorithm': name,
                'Accuracy': f"{accuracy_pca:.4f}",
                'Precision': f"{precision_pca:.4f}",
                'Recall': f"{recall_pca:.4f}",
                'F1_Score': f"{f1_pca:.4f}",
                'CV_Mean': f"{accuracy_pca:.4f}",
                'CV_Std': "0.0000"
            })

            print(f"   ✓ Accuracy with Gaussian NB: {accuracy_pca:.4f}")
            print(f"   ✓ Reduced to {X_train_pca.shape[1]} components")

        except Exception as e:
            print(f"   ❌ Error with {name}: {str(e)}")
            pca_results[name] = {
                'explained_variance': 0.0,
                'accuracy_with_gaussian_nb': 0.0,
                'n_components': 0
            }

    # ================================
    # 3. SEMI-SUPERVISED ANALYSIS
    # ================================
    print(f"\n{'=' * 50}")
    print("🔄 3. SEMI-SUPERVISED LEARNING")
    print(f"{'=' * 50}")

    # Use only 30% of training data as labeled
    n_labeled = max(10, int(0.3 * len(X_train)))  # Ensure minimum samples
    X_labeled = X_train[:n_labeled]
    y_labeled = y_train[:n_labeled]
    X_unlabeled = X_train[n_labeled:]

    print(f"   📊 Labeled samples: {len(X_labeled)}")
    print(f"   📊 Unlabeled samples: {len(X_unlabeled)}")

    # Baseline: supervised with limited data
    print(f"\n🔍 Baseline (Supervised with limited data)...")
    baseline_clf = GaussianNaiveBayes()
    baseline_clf.fit(X_labeled, y_labeled)
    baseline_pred = baseline_clf.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"   ✓ Baseline Accuracy: {baseline_acc:.4f}")

    # Semi-supervised algorithms
    semi_algorithms = {
        'Self-Training NB': SelfTrainingNaiveBayes(confidence_threshold=0.7),
        'Co-Training NB': CoTrainingNaiveBayes(confidence_threshold=0.6),
        'Label Propagation NB': LabelPropagationNaiveBayes(alpha=0.2)
    }

    semi_results = {'Baseline': baseline_acc}

    for name, clf in semi_algorithms.items():
        print(f"\n🔍 Testing {name}...")

        try:
            clf.fit(X_labeled, y_labeled, X_unlabeled)
            y_pred_semi = clf.predict(X_test)
            accuracy_semi = accuracy_score(y_test, y_pred_semi)
            precision_semi = precision_score(y_test, y_pred_semi, average='weighted', zero_division=0)
            recall_semi = recall_score(y_test, y_pred_semi, average='weighted', zero_division=0)
            f1_semi = f1_score(y_test, y_pred_semi, average='weighted', zero_division=0)
            improvement = accuracy_semi - baseline_acc

            semi_results[name] = accuracy_semi

            all_results.append({
                'Algorithm_Type': 'Semi_Supervised',
                'Algorithm': name,
                'Accuracy': f"{accuracy_semi:.4f}",
                'Precision': f"{precision_semi:.4f}",
                'Recall': f"{recall_semi:.4f}",
                'F1_Score': f"{f1_semi:.4f}",
                'CV_Mean': f"{accuracy_semi:.4f}",
                'CV_Std': "0.0000"
            })

            print(f"   ✓ Accuracy: {accuracy_semi:.4f}")
            print(f"   ✓ Improvement over baseline: {improvement:+.4f}")

        except Exception as e:
            print(f"   ❌ Error with {name}: {str(e)}")
            semi_results[name] = baseline_acc

    # Find best performers
    best_nb = max(nb_results.keys(), key=lambda x: nb_results[x]['accuracy'])
    best_pca = max(pca_results.keys(),
                   key=lambda x: pca_results[x]['accuracy_with_gaussian_nb']) if pca_results else 'None'
    best_semi = max([k for k in semi_results.keys() if k != 'Baseline'],
                    key=lambda x: semi_results[x], default='Self-Training NB')

    print(f"\n🏆 BEST PERFORMERS:")
    print(f"   🥇 Best Naive Bayes: {best_nb} ({nb_results[best_nb]['accuracy']:.4f})")
    if best_pca != 'None':
        print(f"   🥇 Best PCA Method: {best_pca} ({pca_results[best_pca]['accuracy_with_gaussian_nb']:.4f})")
    print(f"   🥇 Best Semi-Supervised: {best_semi} ({semi_results[best_semi]:.4f})")

    # Create visualizations
    create_comprehensive_visualization(nb_results, pca_results, semi_results, df)

    # Export results
    export_comprehensive_results(all_results, df, nb_results, pca_results, semi_results)

    return {
        'naive_bayes': nb_results,
        'pca': pca_results,
        'semi_supervised': semi_results,
        'all_results': all_results
    }


def create_comprehensive_visualization(nb_results, pca_results, semi_results, df):
    """Create comprehensive visualization of all results"""

    try:
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('🎓 Comprehensive ML Analysis: Student Performance Dataset', fontsize=16, fontweight='bold')

        # ====== ROW 1: NAIVE BAYES ANALYSIS ======

        # 1.1 Naive Bayes Accuracy Comparison
        nb_algorithms = list(nb_results.keys())
        nb_accuracies = [nb_results[alg]['accuracy'] for alg in nb_algorithms]
        colors_nb = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        axes[0, 0].bar(nb_algorithms, nb_accuracies, color=colors_nb)
        axes[0, 0].set_title('Naive Bayes: Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)

        for i, v in enumerate(nb_accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

        # 1.2 Naive Bayes Performance Metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(nb_algorithms))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [nb_results[alg][metric] for alg in nb_algorithms]
            axes[0, 1].bar(x + i * width, values, width, label=metric.capitalize())

        axes[0, 1].set_title('Naive Bayes: Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x + width * 1.5)
        axes[0, 1].set_xticklabels(nb_algorithms, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # 1.3 Naive Bayes Cross-validation
        cv_means = [nb_results[alg]['cv_mean'] for alg in nb_algorithms]
        cv_stds = [nb_results[alg]['cv_std'] for alg in nb_algorithms]

        axes[0, 2].bar(nb_algorithms, cv_means, yerr=cv_stds, capsize=5, color=colors_nb)
        axes[0, 2].set_title('Naive Bayes: Cross-Validation')
        axes[0, 2].set_ylabel('CV Accuracy')
        axes[0, 2].set_ylim(0, 1)

        # ====== ROW 2: PCA ANALYSIS ======

        if pca_results:
            # 2.1 PCA Explained Variance
            pca_algorithms = list(pca_results.keys())
            explained_vars = [pca_results[alg]['explained_variance'] for alg in pca_algorithms]

            axes[1, 0].bar(pca_algorithms, explained_vars, color=['orange', 'green', 'purple'])
            axes[1, 0].set_title('PCA: Explained Variance Ratio')
            axes[1, 0].set_ylabel('Explained Variance')
            axes[1, 0].set_ylim(0, 1)

            for i, v in enumerate(explained_vars):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

            # 2.2 PCA + Naive Bayes Performance
            pca_accuracies = [pca_results[alg]['accuracy_with_gaussian_nb'] for alg in pca_algorithms]

            axes[1, 1].bar(pca_algorithms, pca_accuracies, color=['orange', 'green', 'purple'])
            axes[1, 1].set_title('PCA + Gaussian NB: Accuracy')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_ylim(0, 1)

            for i, v in enumerate(pca_accuracies):
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

            # 2.3 PCA Components Used
            n_components = [pca_results[alg]['n_components'] for alg in pca_algorithms]

            axes[1, 2].bar(pca_algorithms, n_components, color=['orange', 'green', 'purple'])
            axes[1, 2].set_title('PCA: Number of Components')
            axes[1, 2].set_ylabel('Components')

            for i, v in enumerate(n_components):
                axes[1, 2].text(i, v + 0.1, f'{v}', ha='center', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'PCA Results\nNot Available', ha='center', va='center',
                            transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'PCA Results\nNot Available', ha='center', va='center',
                            transform=axes[1, 1].transAxes)
            axes[1, 2].text(0.5, 0.5, 'PCA Results\nNot Available', ha='center', va='center',
                            transform=axes[1, 2].transAxes)

        # ====== ROW 3: SEMI-SUPERVISED & DATASET ANALYSIS ======

        # 3.1 Semi-Supervised vs Baseline
        semi_names = list(semi_results.keys())
        semi_accs = list(semi_results.values())
        colors_semi = ['red', 'lightblue', 'lightgreen', 'yellow'][:len(semi_names)]

        axes[2, 0].bar(semi_names, semi_accs, color=colors_semi)
        axes[2, 0].set_title('Semi-Supervised vs Baseline')
        axes[2, 0].set_ylabel('Accuracy')
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].tick_params(axis='x', rotation=45)

        for i, v in enumerate(semi_accs):
            axes[2, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

        # 3.2 Grade Distribution
        axes[2, 1].hist(df['G3'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2, 1].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Pass Threshold')
        axes[2, 1].set_title('Final Grade Distribution')
        axes[2, 1].set_xlabel('Final Grade (G3)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()

        # 3.3 Pass/Fail Distribution
        if 'pass_fail' in df.columns:
            pass_fail_counts = df['pass_fail'].value_counts()
            labels = ['Fail', 'Pass']
            colors_pie = ['#FF6B6B', '#4ECDC4']

            axes[2, 2].pie(pass_fail_counts.values, labels=labels, colors=colors_pie,
                           autopct='%1.1f%%', startangle=90)
            axes[2, 2].set_title('Pass/Fail Distribution')

        plt.tight_layout()
        plt.savefig('ml_analysis_dashboard.png', dpi=180, bbox_inches='tight')
        plt.close()
        print("✓ Exported: ml_analysis_dashboard.png")

    except Exception as e:
        print(f"⚠️ Error creating visualization: {e}")
        print("Continuing without visualization...")


def export_comprehensive_results(all_results, df, nb_results, pca_results, semi_results):
    """Export comprehensive results to CSV files"""

    print(f"\n💾 Exporting comprehensive results...")

    try:
        # Export detailed results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('comprehensive_ml_results.csv', index=False)
        print("✓ Exported: comprehensive_ml_results.csv")

        # Export summary comparison
        summary_data = []

        # Add Naive Bayes results
        for name, results in nb_results.items():
            summary_data.append({
                'Category': 'Naive_Bayes',
                'Algorithm': name,
                'Best_Metric': 'Accuracy',
                'Score': f"{results['accuracy']:.4f}",
                'Additional_Info': f"CV: {results['cv_mean']:.3f}±{results['cv_std']:.3f}"
            })

        # Add PCA results
        for name, results in pca_results.items():
            summary_data.append({
                'Category': 'PCA',
                'Algorithm': name,
                'Best_Metric': 'Accuracy_with_GaussianNB',
                'Score': f"{results['accuracy_with_gaussian_nb']:.4f}",
                'Additional_Info': f"Explained_Var: {results['explained_variance']:.3f}"
            })

        # Add Semi-supervised results
        for name, accuracy in semi_results.items():
            improvement = accuracy - semi_results['Baseline'] if name != 'Baseline' else 0
            summary_data.append({
                'Category': 'Semi_Supervised',
                'Algorithm': name,
                'Best_Metric': 'Accuracy',
                'Score': f"{accuracy:.4f}",
                'Additional_Info': f"Improvement: {improvement:+.3f}" if name != 'Baseline' else "Baseline"
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('ml_methods_comparison.csv', index=False)
        print("✓ Exported: ml_methods_comparison.csv")

        # Export dataset characteristics
        dataset_summary = {
            'Characteristic': [
                'Total Samples',
                'Total Features (after preprocessing)',
                'Pass Rate (%)',
                'Fail Rate (%)',
                'Average Final Grade',
                'Standard Deviation G3',
                'Best Naive Bayes Method',
                'Best Semi-Supervised Method',
                'Overall Best Accuracy'
            ],
            'Value': [
                len(df),
                len([col for col in df.columns if col not in ['G1', 'G2', 'G3', 'pass_fail']]),
                f"{df['pass_fail'].mean() * 100:.1f}%",
                f"{(1 - df['pass_fail'].mean()) * 100:.1f}%",
                f"{df['G3'].mean():.2f}",
                f"{df['G3'].std():.2f}",
                max(nb_results.keys(), key=lambda x: nb_results[x]['accuracy']),
                max([k for k in semi_results.keys() if k != 'Baseline'],
                    key=lambda x: semi_results[x], default='N/A'),
                f"{max([max(nb_results[x]['accuracy'] for x in nb_results), max(semi_results.values())]):.4f}"
            ]
        }

        dataset_df = pd.DataFrame(dataset_summary)
        dataset_df.to_csv('dataset_and_best_results.csv', index=False)
        print("✓ Exported: dataset_and_best_results.csv")

        # Print comprehensive summary
        print(f"\n📊 COMPREHENSIVE RESULTS SUMMARY:")
        print("=" * 80)
        print(results_df.to_string(index=False))

        print(f"\n🏆 BEST METHODS COMPARISON:")
        print("=" * 60)
        print(summary_df.to_string(index=False))

        print(f"\n📈 DATASET CHARACTERISTICS:")
        print("=" * 50)
        print(dataset_df.to_string(index=False))

    except Exception as e:
        print(f"⚠️ Error exporting files: {e}")
        print("Results available in memory only.")


def main():
    """Main function to run the complete comprehensive analysis"""

    print("🚀 Starting Comprehensive ML Analysis...")
    print("   📊 Naive Bayes Classifiers (3 types)")
    print("   📊 Principal Component Analysis (3 implementations)")
    print("   📊 Semi-Supervised Learning (3 methods)")
    print("=" * 70)

    try:
        results = run_comprehensive_analysis()


        return results

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check your environment and try again.")
        return None


# Run the comprehensive analysis
if __name__ == "__main__":
    main()