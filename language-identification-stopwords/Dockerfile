FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install langdetect module
RUN pip install tqdm

# Add your script and any other necessary files
ADD language_identification_stopwords.py /code/language_identification_stopwords.py
#ADD stopwords /code/stopwords

# Set the entry point for the container
ENTRYPOINT [ "python3", "/code/language_identification_stopwords.py" ]

