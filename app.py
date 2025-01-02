from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import nltk
import pickle
import numpy as np
from keras.models import load_model
from fpdf import FPDF
import json
import random

app = Flask(__name__)
app.static_folder = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Pastikan folder uploads ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load model dan data NLP
model = load_model('model/models.h5')
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl', 'rb'))
classes = pickle.load(open('model/labels.pkl', 'rb'))

# Untuk menyimpan status pendaftaran pengguna sementara
user_data = {}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/unduh_data')
def unduh_data():
    # Buat PDF secara dinamis
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)

    pdf.cell(200, 30, txt="Data Pendaftaran", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Nama: {user_data.get('name', 'N/A')}", ln=True)
    pdf.cell(200, 10, txt=f"NIP: {user_data.get('nip', 'N/A')}", ln=True)
    pdf.cell(200, 10, txt=f"Tanggal Lahir: {user_data.get('birth', 'N/A')}", ln=True)
    pdf.cell(200, 10, txt=f"Alamat: {user_data.get('address', 'N/A')}", ln=True)
    pdf.cell(200, 10, txt=f"Nomor Telepon: {user_data.get('phone', 'N/A')}", ln=True)
    pdf.cell(200, 10, txt="Berkas Pendaftaran:", ln=True)

    # Menyisipkan gambar jika ada
    images = {
        'akta': user_data.get('akta'),
        'kk': user_data.get('kk'),
        'ijazah': user_data.get('ijazah')
    }

    for label, filename in images.items():
        if filename:
            # Menentukan path gambar
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Memeriksa apakah file gambar ada
            if os.path.exists(img_path):
                pdf.image(img_path, x=10, w=100)  # Mengatur posisi dan lebar gambar
                pdf.ln(5)  # Tambahkan jarak setelah gambar

    # Format nama file yang akan diunduh: "nama_nip_data_pendaftaran.pdf"
    name = user_data.get('name', 'N/A').replace(' ', '_')
    nip = user_data.get('nip', 'N/A')
    pdf_filename = f"{name}_{nip}_data_pendaftaran.pdf"
    
    # Simpan file PDF ke folder uploads
    pdf_output = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    pdf.output(pdf_output)

    # Kembalikan file untuk diunduh dengan nama yang dinamis
    return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/get", methods=['GET', 'POST'])
def get_bot_response():
    userText = request.args.get('msg')

    if userText == "/daftar":
        user_data['step'] = 1
        return "<div>Terima kasih sudah memilih mendaftar di SMP Al Washliyah 9 Medan!<br>Untuk memulai proses pendaftaran, silakan masukan Nama lengkap anda:</div>"

    # Langkah 1: Input nama
    if user_data.get('step') == 1:
        user_data['name'] = userText
        user_data['step'] = 2
        return "<div>Terima kasih, {}. Selanjutnya masukkan NIP:</div>".format(user_data['name'])

    # Langkah 2: Input NIP
    if user_data.get('step') == 2:
        user_data['nip'] = userText
        user_data['step'] = 3
        return "<div>Terima kasih, Sekarang, tolong berikan tanggal lahir kamu (format: DD-MM-YYYY):</div>".format(user_data['nip'])

    # Langkah 3: Input tanggal lahir
    if user_data.get('step') == 3:
        user_data['birth'] = userText
        user_data['step'] = 4
        return "<div>Terima kasih, Selanjutnya, silakan berikan alamat lengkap kamu:</div>".format(user_data['birth'])

    # Langkah 4: Input alamat
    if user_data.get('step') == 4:
        user_data['address'] = userText
        user_data['step'] = 5
        return "<div>Terima kasih! Sekarang, mohon berikan nomor telepon orang tua atau wali yang bisa dihubungi:</div>".format(user_data['address'])

    # Langkah 5: Input nomor telepon
    if user_data.get('step') == 5:
        user_data['phone'] = userText
        user_data['step'] = 6
        return "<div>Terima kasih, Semua informasi sudah diterima. <br>Langkah selanjutnya, silakan unggah Akta kelahiran kamu (dalam format jpg).</div>".format(user_data['phone'])

    # Langkah 6: Upload berkas Akta
    if request.method == 'POST' and user_data.get('step') == 6:
        if 'file' not in request.files:
            return "<div>Tidak ada berkas yang diunggah.</div>"
        file = request.files['file']
        if file.filename == '':
            return "<div>Nama berkas tidak ada.</div>"
        if file and file.filename.lower().endswith(('.jpg', '.jpeg')):
            # Nama berkas Akta
            new_filename = f"{user_data['name'].replace(' ', '_')}_{user_data['nip']}_akta.jpg"
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(new_filepath)

            # Simpan berkas Akta dan lanjut ke KK
            user_data['akta'] = new_filename
            user_data['step'] = 7
            return "<div>Terima kasih, Berkas Akta berhasil diunggah. Sekarang silakan unggah berkas Kartu Keluarga (KK) dalam format JPG.</div>"

    # Langkah 7: Upload berkas KK
    if request.method == 'POST' and user_data.get('step') == 7:
        if 'file' not in request.files:
            return "<div>Tidak ada berkas yang diunggah.</div>"
        file = request.files['file']
        if file.filename == '':
            return "<div>Nama berkas tidak ada.</div>"
        if file and file.filename.lower().endswith(('.jpg', '.jpeg')):
            # Nama berkas KK
            new_filename = f"{user_data['name'].replace(' ', '_')}_{user_data['nip']}_kk.jpg"
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(new_filepath)

            # Simpan berkas KK dan lanjut ke Ijazah
            user_data['kk'] = new_filename
            user_data['step'] = 8
            return "<div>Terima kasih, Berkas Kartu Keluarga (KK) berhasil diunggah. Sekarang silakan unggah berkas Ijazah dalam format JPG.</div>"

    # Langkah 8: Upload berkas Ijazah
    if request.method == 'POST' and user_data.get('step') == 8:
        if 'file' not in request.files:
            return "<div>Tidak ada berkas yang diunggah.</div>"
        file = request.files['file']
        if file.filename == '':
            return "<div>Nama berkas tidak ada.</div>"
        if file and file.filename.lower().endswith(('.jpg', '.jpeg')):
            # Nama berkas Ijazah
            new_filename = f"{user_data['name'].replace(' ', '_')}_{user_data['nip']}_ijazah.jpg"
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(new_filepath)

            # Simpan berkas Ijazah dan selesaikan pendaftaran
            user_data['ijazah'] = new_filename
            user_data['step'] = None  # Reset langkah
            return "<div>Terima kasih, Berkas akta kelahiran telah diterima. Pendaftaran kamu sudah lengkap. Jika kamu ingin melihat data dan berkas yang sudah diunggah, kamu bisa mengetik '/cek_data' atau jika ingin mengunduh data dan berkas pendaftaran ini, kamu bisa mengetik '/unduh_data'.</div>"

    # Menghapus data jika perintah /hapus diberikan
    if userText == "/hapus":
        # Menghapus berkas yang diunggah jika ada
        for key in ['akta', 'kk', 'ijazah']:
            filename = user_data.get(key)
            if filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(filepath):
                    os.remove(filepath)  # Menghapus berkas

        # Menghapus semua data dari dictionary user_data
        user_data.clear()  
        return "<div>Semua data pendaftaran Anda telah dihapus.</div>"

    # Tambahkan logika untuk melihat berkas
    if userText == "/cek_berkas":
        response = "<div>Berikut adalah data dan berkas pendaftaran yang telah kamu unggah:<br>"
        response += f"<div>Nama: {user_data.get('name', 'N/A')}<br>"
        response += f"<div>NIP: {user_data.get('nip', 'N/A')}<br>"
        response += f"<div>Tanggal Lahir: {user_data.get('birth', 'N/A')}<br>"
        response += f"<div>Alamat: {user_data.get('address', 'N/A')}<br>"
        response += f"<div>Nomor Telepon: {user_data.get('phone', 'N/A')}<br>"
        response += "<div>Berkas Pendaftaran<br>"
        
        # Menyisipkan tautan gambar jika ada
        images = {
            '<div>Akta Kelahiran': user_data.get('akta'),
            '<div>Kartu Keluarga': user_data.get('kk'),
            '<div>Ijazah': user_data.get('ijazah')
        }

        for label, filename in images.items():
            if filename:
                img_link = f"<a href='/uploads/{filename}' target='_blank'>{label}</a>"
                response += f"{label}: {img_link}<br>"
        
        response += "</div>"  # Menutup tag div
        return response

    # Tambahkan logika untuk mengunduh data jika pengguna mengetik "/unduh_data"
    if userText == "/unduh_data":
        return "<div><a href='/unduh_data' target='_blank'>Klik <a href='/unduh_data' target='_blank'>di sini</a> untuk mengunduh data Anda</div>"
    
    if userText == "/bantuan":
        return (
            "<div>Berikut adalah daftar perintah yang tersedia:<br>"
            "1. /bantuan - Melihat daftar perintah yang tersedia.<br>"
            "2. /cara_daftar - Mengetahui cara mendaftar di chatbot.<br>"
            "3. /daftar - Memulai proses pendaftaran siswa baru.<br>"
            "4. /cek_berkas - Memeriksa berkas yang sudah diunggah.<br>"
            "5. /unduh_data - Mengunduh data dan berkas pendaftaran.<br><br>"
            "Jika kamu membutuhkan bantuan lebih lanjut, silakan tanyakan!</div>"
        )    
    if userText == "/cara_daftar":
            return (
                "<div>Untuk mendaftar di chatbot, ikuti langkah-langkah berikut:<br>"
                "1. Ketik '/daftar' untuk memulai proses pendaftaran.<br>"
                "2. Chatbot akan meminta informasi berikut:<br>"
                "   - Nama lengkap<br>"
                "   - NIP<br>"
                "   - Tanggal lahir<br>"
                "   - Alamat lengkap<br>"
                "   - Nomor telepon orang tua atau wali<br>"
                "   - Upload berkas Akta Kelahiran, Kartu Keluarga, dan Ijazah dalam format JPG.<br>"
                "3. Setelah mengupload semua berkas, pendaftaran akan dianggap lengkap.<br>"
                "Jika kamu mengalami kesulitan, silakan tanyakan!</div>"
            )

    # Respon default
    ints = predict_class(userText, model)
    res = getResponse(ints, intents)
    return res

if __name__ == "__main__":
    app.run(debug=True)
