FROM python:3.10

# Expose port you want your app on
EXPOSE 8080

#Install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app code and set working directory
COPY app.py app.py
WORKDIR .

# Run
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
