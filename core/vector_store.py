import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
import voyageai
from dotenv import load_dotenv

load_dotenv()


class MedicalVectorStore:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.qdrant_url)
        self.collection_name = "medical_records"
        api_key = os.getenv("VOYAGE_API_KEY")
        if api_key:
            self.voyage_client = voyageai.Client(api_key=api_key)
        else:
            self.voyage_client = None
        self.embedding_model = "voyage-3.5"
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=512,  # Reduced dimension for more general matching
                    distance=models.Distance.COSINE,
                ),
            )
            # Add payload indexes for filterable keys
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="patient_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="event_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="timestamp",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(self, chunks):
        points = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            metadata = chunk["metadata"]

            # Generate embedding with 512 dimensions
            embedding = self.voyage_client.embed(
                [text], model=self.embedding_model, output_dimension=512
            ).embeddings[0]

            # Generate deterministic UUID point ID using internal_id (visit_id, lab_id, etc.)
            # but we don't save this redundant ID in the payload.
            NAMESPACE_MEDICAL = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
            internal_id = chunk.get("internal_id", str(uuid.uuid4()))
            point_id = str(
                uuid.uuid5(
                    NAMESPACE_MEDICAL,
                    f"{metadata['patient_id']}_{internal_id}",
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id, vector=embedding, payload={"text": text, **metadata}
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query, patient_id, limit=5, event_type=None):
        query_embedding = self.voyage_client.embed(
            [query], model=self.embedding_model, output_dimension=512
        ).embeddings[0]

        must_filters = [
            models.FieldCondition(
                key="patient_id", match=models.MatchValue(value=patient_id)
            )
        ]

        if event_type:
            must_filters.append(
                models.FieldCondition(
                    key="event_type", match=models.MatchValue(value=event_type)
                )
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(must=must_filters),
            limit=limit,
        )

        return results
