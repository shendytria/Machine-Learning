from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, os, uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

RUNS = {}

# ===================== Helper Functions =====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def decode_features(dfX: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df_dec = dfX.copy()
    for col, le in encoders.items():
        if col in df_dec.columns:
            arr = pd.to_numeric(df_dec[col], errors='coerce').fillna(0).astype(int)
            df_dec[col] = le.inverse_transform(arr)
    return df_dec


# ===================== Routes =====================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Please upload a valid CSV file'}), 400

        df = pd.read_csv(file)
        preview_html = df.to_html(classes='table table-striped', index=False)

        # Filter kolom yang cocok jadi target
        target_candidates = []
        for col in df.columns:
            unique_count = df[col].nunique(dropna=True)
            dtype = df[col].dtype
            if dtype == 'object':
                target_candidates.append(col)
            elif unique_count <= 10:
                target_candidates.append(col)

        return jsonify({
            'success': True,
            'columns': df.columns.tolist(),
            'target_candidates': target_candidates,
            'rows': len(df),
            'preview': preview_html
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    try:
        file = request.files['file']
        target_column = request.form.get('target_column')
        test_size = float(request.form.get('test_size', 0.2))

        df = pd.read_csv(file)
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found'}), 400

        df = df[df[target_column].notna()].copy()
        df[target_column] = df[target_column].astype(str).str.strip()
        df = df[df[target_column].str.lower() != 'nan']

        X_raw = df.drop(columns=[target_column]).copy()
        y_raw = df[target_column].copy()

        encoders = {}
        feature_meta = []
        X = X_raw.copy()

        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
                feature_meta.append({'name': col, 'type': 'categorical', 'categories': le.classes_.tolist()})
            else:
                try:
                    X[col] = pd.to_numeric(X[col], errors='raise')
                    feature_meta.append({'name': col, 'type': 'numeric'})
                except:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le
                    feature_meta.append({'name': col, 'type': 'categorical', 'categories': le.classes_.tolist()})

        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        kernels = {
            'linear': {'kernel': 'linear'},
            'polynomial': {'kernel': 'poly', 'degree': 3},
            'rbf': {'kernel': 'rbf', 'gamma': 'scale'}
        }

        results = {}
        models_store = {}

        y_train_labels = target_encoder.inverse_transform(y_train)
        y_test_labels = target_encoder.inverse_transform(y_test)
        X_train_decoded = decode_features(pd.DataFrame(X_train, columns=X.columns), encoders)
        X_test_decoded = decode_features(pd.DataFrame(X_test, columns=X.columns), encoders)

        def build_train_table_html():
            df_disp = X_train_decoded.copy()
            df_disp[target_column] = pd.Series(y_train_labels)
            return df_disp.to_html(classes='table table-sm table-striped', index=False)

        def build_test_table_html(y_pred_labels):
            df_disp = X_test_decoded.copy()
            df_disp[target_column] = pd.Series(y_test_labels)
            # y_pred disembunyikan
            return df_disp.to_html(classes='table table-sm table-striped', index=False)

        # Train tiap kernel
        for kname, kparams in kernels.items():
            model = SVC(**kparams, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_labels = target_encoder.inverse_transform(y_pred)

            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            avg = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
            cm_plot = plot_confusion_matrix(cm, f'Confusion Matrix - {kname.upper()} Kernel')

            results[kname] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'cm_plot': cm_plot
            }

            models_store[kname] = model

        # Simpan run
        run_id = str(uuid.uuid4())
        RUNS[run_id] = {
            'feature_names': X.columns.tolist(),
            'target_name': target_column,
            'scaler': scaler,
            'encoders': encoders,
            'target_encoder': target_encoder,
            'models': models_store,
            'feature_meta': feature_meta
        }

        train_table_html = build_train_table_html()
        test_table_html = build_test_table_html(target_encoder.inverse_transform(y_test))

        return jsonify({
            'success': True,
            'run_id': run_id,
            'results': results,
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X.columns.tolist(),
                'target': target_column,
                'classes': int(len(np.unique(y)))
            },
            'feature_meta': feature_meta,
            'train_table': train_table_html,
            'test_table': test_table_html
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.get_json(force=True)
        run_id = data.get('run_id')
        kernel = data.get('kernel')
        feats = data.get('features', {})

        if run_id not in RUNS:
            return jsonify({'error': 'Invalid run_id'}), 400
        ctx = RUNS[run_id]
        if kernel not in ctx['models']:
            return jsonify({'error': 'Invalid kernel'}), 400

        feature_names = ctx['feature_names']
        encoders = ctx['encoders']
        scaler = ctx['scaler']
        target_encoder = ctx['target_encoder']

        row = {}
        for name in feature_names:
            val = feats.get(name)
            if name in encoders:
                val_str = str(val)
                if val_str not in encoders[name].classes_:
                    return jsonify({'error': f'Unknown category for "{name}": "{val_str}"'}), 400
                row[name] = encoders[name].transform([val_str])[0]
            else:
                row[name] = float(val)

        X_manual = pd.DataFrame([row], columns=feature_names)
        X_scaled = scaler.transform(X_manual)

        model = ctx['models'][kernel]
        pred = model.predict(X_scaled)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]

        return jsonify({'success': True, 'predicted': str(pred_label)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
