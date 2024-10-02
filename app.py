import threading
from flask import Flask, render_template_string, request, jsonify, render_template, send_from_directory
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ultralytics import YOLO
import cv2
import datetime
import time
import platform
from flask import Flask, request, jsonify, render_template
from langchain_utils import LangChainHelper #Langchain_utils.py에서 구현된 클래스이며, PDF 파일을 처리하고 질문에 대한 답변을 생성하는 역할
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import random


# 침입 감지 시 소리 재생 (Windows 전용)
if platform.system() == "Windows":
    import winsound

# YOLOv8s 모델 로드
model = YOLO("yolov8s.pt")

# 침입 로그 리스트 및 상태
intrusion_log = []
intrusion_detected = False  # 침입 상태를 추적하기 위한 bool 플래그
is_running = False

# 웹캠 피드 열기
cap = cv2.VideoCapture(0)

# Flask 애플리케이션 설정
app = Flask(__name__)

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/yolo')
def yolo():
    # 웹페이지에 로그 및 침입 상태 전달
    return render_template('yolo.html', logs=intrusion_log, intrusion=intrusion_detected)

@app.route('/intrusion_status')
def intrusion_status():
    # 현재 침입 상태를 JSON으로 반환
    return jsonify({"intrusion": intrusion_detected})

@app.route('/start_detection')
def start_detection():
    global is_running
    is_running = True
    return jsonify({"status": "Detection started"})

@app.route('/stop_detection')
def stop_detection():
    global is_running
    is_running = False
    return jsonify({"status": "Detection stopped"})

@app.route('/get_logs')
def get_logs():
    # 칩입 로그를 JSON으로 반환
    return jsonify({"logs": intrusion_log})

def detect_people():
    global intrusion_log, intrusion_detected, is_running

    while True:
        if not is_running:
            if cap.isOpened():
                cap.release()
            intrusion_detected = False 
            time.sleep(1)
            continue

        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            cap.open(0)

        # if not cap.isOpened():
        #     cap.open(0)
            
        try:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 가져오지 못했습니다.")
                break

            # 사람 탐지를 위해 YOLO 모델 실행
            # results = model.predict(source=frame, show=True)

            frame_resized = cv2.resize(frame, (640, 640))
            results = model(frame)

            # 각 프레임마다 침입 상태 초기화
            intrusion_detected = False

            # 사람이 감지되었는지 확인 (COCO 클래스 ID 0)
            for result in results:
                for r in result.boxes.data:
                    class_id = int(r[-1])
                    if class_id == 0:  # 사람이 감지된 경우
                        intrusion_detected = True
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"침입 감지: {timestamp}"
                        intrusion_log.append(log_entry)

                        # # 마지막 10개의 로그만 유지
                        # if len(intrusion_log) > 10:
                        #     intrusion_log.pop(0)

                        # 침입 감지 시 경고음 재생
                        if platform.system() == "Windows":
                            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                        elif platform.system() == "Darwin":  # MacOS
                            os.system('afplay /System/Library/Sounds/Glass.aiff')


            time.sleep(1)  # CPU 사용량을 줄이기 위해 약간의 지연 추가
        except Exception as e:
            print(f"YOLO 감지 중 오류: {e}")





# CNN 모델 로드
model_cnn = load_model('./cnn_model.h5')

# 이미지 저장 경로
IMAGE_FOLDER = '../cat_dog'

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 리스트 API
@app.route('/images', methods=['GET'])
def get_images():
    images = []
    for label in ['cat', 'dog']:
        label_folder = os.path.join(IMAGE_FOLDER, label)
        for filename in os.listdir(label_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                images.append({'src': f'/images/{label}/{filename}', 'label': label})
    
    random.shuffle(images)
    return jsonify(images[:16])  # 16개 이미지 반환

# 이미지 파일 제공
@app.route('/images/<label>/<filename>')
def serve_image(label, filename):
    return send_from_directory(os.path.join(IMAGE_FOLDER, label), filename)

# 이미지 예측 API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    # file_path = f'/tmp/{file.filename}'
    # file.save(file_path)
    temp_dir = 'C:/temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)


    # 이미지 전처리
    img = image.load_img(file_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    # 예측 수행
    prediction = model_cnn.predict(img)
    label = 'dog' if prediction[0][0] >= 0.5 else 'cat'

    os.remove(file_path)

    return jsonify({'prediction': label})




# 업로드 폴더 생성 확인
# PDF 파일을 업로드하기 전에 uploads 폴더가 존재하는지 확인하고, 존재하지 않으면 폴더를 생성
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# LangChainHelper 인스턴스 생성 (클래스를 초기화하여 lc_helper 객체를 생성, 이 객체는 PDF 파일을 처리하고 질문에 대한 답변을 생성에 활용)
lc_helper = LangChainHelper()

@app.route('/langchain')
def langchain_page():
    return render_template('langchain.html')


#  PDF 파일 업로드 API (사용자가 PDF 파일을 업로드하면 이 API가 호출)
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "파일을 선택해주세요."}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    lc_helper.process_pdf(pdf_file)   # LangChainHelper를 사용하여 업로드된 PDF 파일을 처리. PDF 파일을 텍스트로 변환하고 벡터로 임베딩하여 벡터 스토어에 저장
    return jsonify({"message": "PDF 파일이 성공적으로 업로드되었습니다."})


# 질문을 통해 답변을 받는 API (사용자가 질문을 입력하면 해당 API가 호출)
@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get('question') # POST 요청으로 전달된 JSON 데이터에서 question 키의 값을 가지고 온다.
    if not question:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    answer = lc_helper.get_answer(question) #LangChainHelper에서 질문에 대해 생성한 답변을 가지고 온다.
    return jsonify({"answer": answer})




# ChromaDB 클라이언트 설정
client = chromadb.Client()
collection = client.create_collection("security_manual")

# 파인튜닝된 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("kingkim/kodialogpt_v3.0_SecurityManual")
model_gpt = AutoModelForCausalLM.from_pretrained("kingkim/kodialogpt_v3.0_SecurityManual")
gpt2_pipeline = pipeline('text-generation', model=model_gpt, tokenizer=tokenizer)

# SentenceTransformer 모델 로드 (임베딩 모델을 더 성능이 좋은 paraphrase-multilingual로 교체)
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 1. PDF 업로드 및 처리 (문맥을 포함한 임베딩을 위해 세 문장씩 묶음)
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    sentences = text.split('.')
    grouped_sentences = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    return grouped_sentences

# 2. 문장 임베딩 후 ChromaDB에 저장
def store_in_chroma(sentences):
    embeddings = sentence_model.encode(sentences)

    # 각 임베딩을 리스트로 변환
    embeddings = [embedding.tolist() for embedding in embeddings]

    ids = [f"id_{i}" for i in range(len(sentences))]  # 각 문장에 대한 고유 ID 생성
    for i, sentence in enumerate(sentences):
        collection.add(ids=[ids[i]], embeddings=[embeddings[i]], metadatas={"sentence": sentence})

        
# 3. 질문 인코딩 및 유사 문장 찾기 (상위 3개의 결과를 결합하여 사용)
def find_similar_sentences(question, top_n=3):
    question_embedding = sentence_model.encode([question])
    results = collection.query(query_embeddings=question_embedding, n_results=top_n)
    
    similar_sentences = []
    if "metadatas" in results:
        for metadata in results["metadatas"]:
            similar_sentences.append(metadata[0].get("sentence", ""))
    
    return ' '.join(similar_sentences)  # 가장 유사한 문장들을 결합하여 반환

# 4. GPT-2 모델로 답변 생성 (질문을 포함한 프롬프트 추가)
def generate_response(similar_sentence, question):
    prompt = f"질문: {question}\n유사한 문장: {similar_sentence}\n답변:"
    generation_args = {
        "max_new_tokens": 80,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 2,
        "early_stopping": True
    }
    response = gpt2_pipeline(prompt, **generation_args)
    return response[0]['generated_text']

# 5. Flask 라우팅 설정
@app.route('/economy_chatbot')
def economy_chatbot():
    return render_template('tuned.html')

@app.route('/eco_upload_pdf', methods=['POST'])
def eco_upload_pdf():
    file = request.files['pdf']
    if file:
        file.save('uploaded_pdf.pdf')
        sentences = process_pdf('uploaded_pdf.pdf')
        store_in_chroma(sentences)
        return jsonify({"message": "PDF 업로드 및 처리 완료"}), 200
    return jsonify({"message": "파일 업로드 실패"}), 400

@app.route('/eco_ask_question', methods=['POST'])
def eco_ask_question():
    data = request.get_json()
    question = data.get('question', '')
    if question:
        similar_sentence = find_similar_sentences(question)
        answer = generate_response(similar_sentence, question)
        return jsonify({"answer": answer}), 200
    return jsonify({"answer": "질문을 입력해주세요"}), 400







# Flask 서버를 백그라운드 스레드에서 실행
def run_flask():
    print("Flask 서버 시작 중...")
    app.run(debug=False, use_reloader=False)    


# YOLO 감지 스레드 실행
yolo_thread = threading.Thread(target=detect_people)
yolo_thread.daemon = True
yolo_thread.start()
print("YOLO 감지 스레드가 시작되었습니다.")

# Flask 서버 스레드 실행
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()
print("Flask 서버 스레드가 시작되었습니다.")

# 메인 스레드 유지
while True:
    time.sleep(1)
