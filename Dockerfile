FROM python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]