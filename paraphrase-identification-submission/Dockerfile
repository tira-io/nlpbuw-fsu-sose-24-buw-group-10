FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip install sentence-transformers

ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]