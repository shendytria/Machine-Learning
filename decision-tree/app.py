from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from graphviz import Source

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Global (DEV-only) storage
global_model = None
global_feature_names = None
global_target_column = None
global_original_columns = None
global_raw_df = None
global_df_processed = None
global_X_train_sample = None
global_X_test_sample = None
global_y_train_sample = None
global_y_test_sample = None
global_y_pred = None
global_accuracy = None
global_class_report = None
global_confusion_matrix = None
global_tree_base64 = None

# Helper: convert matplotlib figure to base64 string
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global global_model, global_feature_names, global_target_column, global_original_columns
    global global_raw_df, global_df_processed
    global global_X_train_sample, global_X_test_sample, global_y_train_sample, global_y_test_sample
    global global_y_pred, global_accuracy, global_class_report, global_confusion_matrix
    global global_tree_base64

    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    try:
        # read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return "Unsupported file type. Use .csv or .xlsx", 400

        if df.shape[1] < 2:
            return "Dataset must have at least 2 columns (features + target).", 400

        # save raw df and metadata
        global_raw_df = df.copy()
        global_original_columns = df.columns.tolist()
        global_target_column = df.columns[-1]  # assume last column is target

        # ---- Data awal (sample) ----
        data_head_html = df.head(10).to_html(classes='table table-striped', index=False)

        # ---- Grouping variabel: X and y ----
        X_df = df.drop(columns=[global_target_column])
        y_df = df[[global_target_column]]
        X_html = X_df.head(10).to_html(classes='table table-striped', index=False)
        y_html = y_df.head(10).to_html(classes='table table-striped', index=False)

        # ---- Preprocessing: get_dummies to handle categorical variables ----
        # Keep a processed copy to ensure consistent encoding later
        df_processed = pd.get_dummies(X_df, drop_first=True)
        global_df_processed = df_processed

        X = df_processed
        y = df[global_target_column]

        # ---- Split 80-20 (train 80% / test 20%) ----
        # try to stratify if possible
        stratify_arg = y if len(y.unique()) > 1 and len(y) >= (len(y.unique()) * 2) else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

        # Save sample HTML tables (sample rows)
        # Re-map back to original columns where possible for display clarity:
        # We'll show the processed features for samples (since model uses them).
        X_train_sample_html = pd.DataFrame(X_train).head(10).to_html(classes='table table-striped', index=False)
        X_test_sample_html = pd.DataFrame(X_test).head(10).to_html(classes='table table-striped', index=False)
        y_train_sample_html = pd.DataFrame(y_train).head(10).to_html(classes='table table-striped', index=False)
        y_test_sample_html = pd.DataFrame(y_test).head(10).to_html(classes='table table-striped', index=False)
       
        # ---- Train Decision Tree ----
        dt_classifier = DecisionTreeClassifier(max_depth=6, random_state=42)
        dt_classifier.fit(X_train, y_train)

        global_model = dt_classifier
        global_feature_names = X.columns.tolist()

        # ---- Predict on X_test ----
        y_pred = dt_classifier.predict(X_test)
        global_y_pred = y_pred

        y_pred_html = pd.DataFrame({'Predicted (y_pred)': y_pred[:10]}).to_html(
        classes='table table-striped', index=False
    )

        # ---- Metrics ----
        acc = accuracy_score(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        global_accuracy = acc
        global_class_report = class_rep
        global_confusion_matrix = cm

        # ---- Confusion matrix heatmap (base64) ----
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        cm_b64 = fig_to_base64(fig)

        # ---- Optional: export tree (graphviz) to PNG base64 (may fail if graphviz not installed)
        tree_b64 = None
        try:
            dot_data = export_graphviz(
                dt_classifier,
                out_file=None,
                feature_names=global_feature_names,
                class_names=[str(c) for c in dt_classifier.classes_],
                filled=True, rounded=True, special_characters=True
            )
            graph = Source(dot_data, format="png")
            png_output = graph.pipe(format='png')
            tree_b64 = base64.b64encode(png_output).decode('utf-8')
        except Exception as e:
            # graphviz may not be available; not fatal
            tree_b64 = None

        # store HTML snippets and other results into global vars to pass to template
        global_X_train_sample = X_train_sample_html
        global_X_test_sample = X_test_sample_html
        global_y_train_sample = y_train_sample_html
        global_y_test_sample = y_test_sample_html
        global_tree_base64 = tree_b64

        # Render data page
        return render_template(
            'data.html',
            data_head_html=data_head_html,
            X_html=X_html,
            y_html=y_html,
            X_train_sample_html=X_train_sample_html,
            X_test_sample_html=X_test_sample_html,
            y_train_sample_html=y_train_sample_html,
            y_test_sample_html=y_test_sample_html,
            y_pred_html=y_pred_html,
            accuracy=round(acc * 100, 2),
            class_report=class_rep,
            confusion_matrix=cm.tolist(),
            confusion_matrix_img=cm_b64,
            tree_img=tree_b64,
            columns=global_original_columns,
            target_column=global_target_column
        )

    except Exception as e:
        return f"Error processing file: {e}", 500

@app.route('/predict', methods=['GET', 'POST'])
def predict_input():
    global global_model, global_feature_names, global_target_column, global_original_columns, global_raw_df, global_df_processed

    if global_model is None:
        return redirect(url_for('index'))

    # Ambil kolom fitur dari dataset asli (tanpa target)
    feature_columns = [c for c in global_original_columns if c != global_target_column]

    # Bentuk pertanyaan otomatis dari nama kolom
    questions = [(col, f"Masukkan nilai untuk '{col}':") for col in feature_columns]

    prediction_result = None
    input_values = {}

    if request.method == 'POST':
        # Kumpulkan input user
        input_data = {}
        for col in feature_columns:
            raw = request.form.get(f'input_{col}', '')
            try:
                val = float(raw)
                if val.is_integer():
                    val = int(val)
            except:
                val = raw.strip()
            input_data[col] = val

        # Buat dataframe dari input
        input_df = pd.DataFrame([input_data])

        # Gabungkan dengan data training → biar konsisten one-hot
        train_features_df = global_raw_df.drop(columns=[global_target_column])
        combined_raw = pd.concat([input_df, train_features_df], ignore_index=True, sort=False)
        combined_encoded = pd.get_dummies(combined_raw, drop_first=True)

        # Reindex ke feature names model
        final_input = combined_encoded.iloc[0].reindex(index=global_feature_names, fill_value=0).values.reshape(1, -1)

        # Prediksi
        prediction = global_model.predict(final_input)[0]
        prediction_result = str(prediction)
        input_values = input_data

    # Render template dengan pertanyaan
    return render_template(
        'predict.html',
        questions=questions,
        target_column=global_target_column,
        prediction_result=prediction_result,
        input_values=input_values
    )

if __name__ == '__main__':
    app.run(debug=True)
