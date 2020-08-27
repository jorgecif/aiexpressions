# Core pkgs
import streamlit as st
import imageio
import io


# EDA pkgs
import pandas as pd 
import numpy as np
import os
import sys
import base64
from pandas import DataFrame
import time



# Data visulization pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
st.set_option('deprecation.showfileUploaderEncoding', False)  # Apagar warning

# ML Pkgs
import cv2
from fer import FER
from fer import Video



def main():
	"""Análisis de sentimientos"""

	st.title("Aplicación de aprendizaje automático")
	st.text("Análisis de sentimiento en fotos y videos")

	activites = ["Imagen","Video"]

	choice = st.sidebar.selectbox("Seleccione",activites)

	if choice == 'Imagen':
		st.subheader("Análisis de imagen")
		uploaded_file = st.file_uploader("Subir imagen",type=["png","jpg"])
		if uploaded_file is not None:

			file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
			opencv_image = cv2.imdecode(file_bytes, 1)
			st.image(opencv_image, channels="BGR", use_column_width=True)
			detector = FER()
			#detector = FER(mtcnn=True)
			result = detector.detect_emotions(opencv_image)
			bounding_box = result[0]["box"]
			emotions = result[0]["emotions"]
			result_df=DataFrame(result)
			result_dic=result_df.loc[: , "emotions"][0]
			result_dic_df=pd.Series(result_dic).to_frame(name="Score")

			my_bar = st.progress(0)

			for percent_complete in range(100):
				time.sleep(0.1)
				my_bar.progress(percent_complete + 1)

			cv2.rectangle(
				opencv_image,
				(bounding_box[0], bounding_box[1]),
				(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
				(0, 155, 255),
				2,
			)
			for idx, (emotion, score) in enumerate(emotions.items()):
				color = (211, 211, 211) if score < 0.01 else (0, 255, 0)
				emotion_score = "{}: {}".format(
					emotion, "{:.2f}".format(score) if score > 0.01 else ""
				)
				cv2.putText(
					opencv_image,
					emotion_score,
					(bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + idx * 15),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					color,
					1,
					cv2.LINE_AA,
				)
			cv2.imwrite("imagen_result.jpg", opencv_image)
			st.subheader('Imagen analizada')
			st.image(opencv_image, channels="BGR", use_column_width=True)
			st.markdown(get_binary_file_downloader_html("imagen_result.jpg", 'Imagen'), unsafe_allow_html=True)

			emocion, puntaje = detector.top_emotion(opencv_image) 
			st.write("Emoción predominante: ", emocion)
			st.write("Puntaje: ", puntaje)

			st.markdown("---")
			st.subheader('Tabla de resultados')
			st.table(result_dic_df)
			st.subheader('Gráfica')
			st.line_chart(result_dic_df)



	elif choice == 'Video':
		st.subheader("Análisis de video")
		uploaded_video = st.file_uploader("Subir video en .mp4",type=["mp4"])

		if uploaded_video is not None:
			video_bytes = np.asarray(bytearray(uploaded_video.read()), dtype=np.uint8)
			st.video(video_bytes)
			#video_path = st.text_input('CSV file path')

			temporary_location = "video_result.mp4"
			with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
				out.write(video_bytes)  ## Read bytes into file
			out.close()
			video = Video("video_result.mp4")
			# Analyze video, displaying the output
			detector = FER()
			raw_data = video.analyze(detector, display=True)
			
			
			# Barra progreso
			my_bar = st.progress(0)
			for percent_complete in range(100):
				time.sleep(0.1)
				my_bar.progress(percent_complete + 1)
			st.markdown("---")
			st.subheader('Tabla de resultados')			
			# Resultados			
			df = video.to_pandas(raw_data)
			st.table(df.head())
			# Plot emotions
			medias=df[['angry','disgust','fear','happy','sad','surprise','neutral']].mean()
			st.write(medias)
			st.bar_chart(medias)
			st.subheader('Gráfica')	
			st.line_chart(df[['angry','disgust','fear','happy','sad','surprise','neutral']])
	
	elif choice == 'Créditos':
		st.subheader("Créditos")
		st.text("Jorge O. Cifuentes")
		st.write('*jorgecif@gmail.com* :sunglasses:')

# Función para bajar archivo generado después del análisis
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Descargar {file_label}</a>'
    return href


if __name__ == "__main__":
    main()