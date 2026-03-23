from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
import uuid

from src.core.config import settings
from src.core.logger import logger


class Indexer:
    """
    Handles connection and data ingestion into the Qdrant vector database.
    """

    def __init__(self):
        self.url = settings.QDRANT_URL
        self.client  = QdrantClient(url=self.url)
        self.collection_name = settings.config_yaml.get('qdrant').get("collection_name", "rag")
        self.distance = settings.config_yaml.get('qdrant').get("distance", "COSINE")
        self.dimensions = settings.config_yaml.get('embedder').get("dimensions", 1024)

    def check_collection_existence(self):
        """
        Verifies if the target Qdrant collection exists, creating it with appropriate Cosine/Dimension
        vector configurations if it does not
        """

        try:
            exists = self.client.collection_exists(collection_name=self.collection_name)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimensions,
                        distance=Distance[self.distance.upper()]
                    )
                )
                logger.info(
                    "Collection does not exist. Creating a new one...", 
                    collection_name=self.collection_name
                )
            else:
                logger.info("Collection already existed.")
        except Exception as e:
            logger.error("Failed to check collection existence", error=str(e))
            raise e

    def index_documents(self, chunks: list[dict]): 
        """
        Transforms raw chunk dictionaries into Qdrant PointStructs and upserts them into the database.
        """

        try:
            points = []
            for c in chunks:
                vector = c.get('embedding')
                text = c.get("text")
                metadata = c.get("metadata")

                source = metadata.get("source")
                text_to_encode = f"{source}||{text}"
                id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text_to_encode))

                metadata["text"] = text
                payload=metadata

                points.append(PointStruct(
                    vector=vector,
                    payload=payload,
                    id=id
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points,
            )
            logger.info("Successfully indexed documents to Qdrant", count=len(points))

        except Exception as e:
            logger.error("Failed to index documents to Qdrant", error=str(e))
            raise e



    
