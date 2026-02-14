from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetectorDLModel, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter
from pyspark.ml import Pipeline
from pyspark.sql.functions import explode
import sparknlp
from pyspark.sql import SparkSession

# Stop existing Spark Session if it exists
if 'spark' in locals() and spark is not None:
    spark.stop()

# Get Spark NLP version for package configuration
spark_nlp_version = sparknlp.version()

# Explicitly build SparkSession with Spark NLP package
spark = SparkSession.builder \
    .appName("Spark NLP") \
    .master("local[*]") \
    .config("spark.driver.memory", "16G") \
    .config("spark.executor.memory", "16G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:" + spark_nlp_version) \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .getOrCreate()

print("Spark NLP version: ", spark_nlp_version)
print("Apache Spark version: ", spark.version)
print("Spark NLP and SparkSession (re)initialized successfully.")


# Sample clinical text for entity extraction
clinical_text = "The patient presented with chest pain, fever, and a persistent cough. Diagnosis included pneumonia and possible bronchitis. Prescribed amoxicillin 500mg daily."

# Step 1: Document Assembler - Transforms raw text into a document annotation
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

# Step 2: Sentence Detector - Splits the document into sentences
sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# Step 3: Tokenizer - Splits sentences into tokens
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

# Step 4: Word Embeddings - Uses pre-trained word embeddings for general text
word_embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

# Step 5: NER Model - Recognizes general entities in text (using a medical-like dataset for demo)
ner_model = NerDLModel.pretrained("onto_recognize_entities_sm", "en") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

# Step 6: NerConverter - Converts NER annotations into a more readable format
ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

# Assemble the NLP pipeline
nlp_pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    word_embeddings,
    ner_model,
    ner_converter
])

# Create a Spark DataFrame for the clinical text
data = spark.createDataFrame([[clinical_text]]).toDF("text")

# Run the pipeline
result = nlp_pipeline.fit(data).transform(data)

# Extract and display the medical entities in a readable format
extracted_entities = result.select(explode("ner_chunk").alias("ner_chunk_item")) \
    .select(
        "ner_chunk_item.result",
        "ner_chunk_item.metadata.entity"
    ) \
    .collect()

print("Extracted Medical Entities:")
for item in extracted_entities:
    print(f"  - Entity: {item['result']}, Type: {item['entity']}")