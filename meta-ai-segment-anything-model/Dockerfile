FROM python:3.9

WORKDIR /app

COPY . /app
COPY notebooks/ /app/notebooks/

RUN pip install --upgrade pip

RUN apt-get update

RUN apt-get install -y libgdal-dev

RUN pip install GDAL==3.6.2

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1128

# Define environment variable
ENV NAME World

# Run when the container launches
CMD ["jupyter", "notebook", "--ip='*'", "--port=1128", "--no-browser", "--allow-root"]