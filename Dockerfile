FROM python:3.7

# Expose port you want your app on
EXPOSE 8080

#Copy local code to the container image
WORKDIR .
COPY ../

# Install production dependencies
RUN pip install -r requirements.txt


CMD streamlit run --server.port 8080 --server.enableCORS false app.py
