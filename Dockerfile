# pull python image
FROM python:3.12.1

# add app files
ADD . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["python", "api/app.py"]

