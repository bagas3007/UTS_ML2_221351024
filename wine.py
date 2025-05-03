import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os

# Judul aplikasi
st.title("Prediksi Kualitas Wine")

# Fungsi untuk memuat model dan scaler
def load_assets():
    try:
        # Cek file model
        if not os.path.exists('wine_quality_model.tflite'):
            st.warning("File model 'wine_quality_model.tflite' tidak ditemukan. Mode demo diaktifkan.")
            return None, None
        
        # Cek file scaler
        if not os.path.exists('scaler.pkl'):
            st.warning("File scaler 'scaler.pkl' tidak ditemukan. Mode demo diaktifkan.")
            return None, None
        
        # Load model TFLite
        interpreter = tf.lite.Interpreter(model_path='wine_quality_model.tflite')
        interpreter.allocate_tensors()
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return interpreter, scaler
        
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return None, None

# Fungsi prediksi
def predict_quality(interpreter, scaler, input_data):
    # Preprocessing
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array).astype(np.float32)
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_score = output[0][0]
    
    # Klasifikasi kualitas
    if pred_score >= 0.7:
        return "Sangat Berkualitas", pred_score
    elif pred_score >= 0.5:
        return "Berkualitas", pred_score
    else:
        return "Biasa", pred_score

# Simulasi prediksi untuk mode demo
def demo_prediction():
    import random
    score = random.uniform(0.3, 0.9)
    if score >= 0.7:
        return "Sangat Berkualitas", score
    elif score >= 0.5:
        return "Berkualitas", score
    else:
        return "Biasa", score

# Memuat model dan scaler
interpreter, scaler = load_assets()

# Cek mode (demo atau normal)
demo_mode = (interpreter is None or scaler is None)
if demo_mode:
    st.info("Aplikasi berjalan dalam mode DEMO karena model tidak tersedia.")

# Input fitur
st.subheader("Masukkan Nilai Fitur Wine:")
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.0, 0.7, step=0.01)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9, step=0.1)
    chlorides = st.number_input("Chlorides", 0.01, 0.5, 0.076, step=0.001)
    
with col2:
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1, 100, 11, step=1)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 5, 300, 34, step=1)
    density = st.number_input("Density", 0.98, 1.05, 0.9978, step=0.0001)
    pH = st.number_input("pH", 2.5, 4.5, 3.51, step=0.01)
    sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56, step=0.01)
    alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4, step=0.1)

# Tombol prediksi
if st.button("Prediksi Kualitas"):
    input_features = [
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]
    
    try:
        if demo_mode:
            # Mode demo
            quality, score = demo_prediction()
            label = "(DEMO)"
        else:
            # Mode normal
            quality, score = predict_quality(interpreter, scaler, input_features)
            label = ""
            
        # Tampilkan hasil
        st.success(f"**Hasil Prediksi {label}:** {quality}")
        st.info(f"**Skor Prediksi {label}:** {score:.1f}")
        
        # Visualisasi
        st.progress(float(score))
        st.metric("Tingkat Kualitas", quality, f"{score*100:.2f}%")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

# Informasi tambahan
st.markdown("""
**Panduan Penggunaan:**
1. Isi semua parameter wine
2. Klik tombol "Prediksi Kualitas"
3. Hasil akan muncul beserta skor prediksi

**Klasifikasi Kualitas:**
- Skor ≥ 0.7: Sangat Berkualitas
- Skor ≥ 0.5: Berkualitas
- Skor < 0.5: Biasa
""")

# Footer
st.sidebar.header("Tentang Aplikasi")
st.sidebar.markdown("""
Aplikasi Prediksi Kualitas Wine ini menggunakan model machine learning 
untuk mengevaluasi kualitas wine berdasarkan karakteristik kimianya.

© 2025 Wine Quality Predictor
""")

# Informasi versi (opsional)
try:
    st.sidebar.markdown(f"""
    **Info Versi:**
    - TensorFlow: {tf.__version__}
    - NumPy: {np.__version__}
    - Streamlit: {st.__version__}
    """)
except:
    pass