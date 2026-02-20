from types import SimpleNamespace

from router.app.qdrant_schema import COLLECTION_NAME, ensure_payload_indexes, ensure_schema


class FakeClient:
    def __init__(self) -> None:
        self.collections = {}
        self.created_indexes = []
        self.last_vectors_config = None

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise RuntimeError("collection not found")
        return self.collections[collection_name]

    def create_collection(self, collection_name: str, vectors_config=None):
        self.last_vectors_config = vectors_config
        self.collections[collection_name] = SimpleNamespace(payload_schema={})

    def create_payload_index(self, collection_name: str, field_name: str, field_schema):
        collection = self.collections[collection_name]
        collection.payload_schema[field_name] = field_schema
        self.created_indexes.append(field_name)


def test_ensure_schema_and_indexes_idempotent() -> None:
    client = FakeClient()

    ensure_schema(client)
    assert COLLECTION_NAME in client.collections
    assert client.last_vectors_config is not None

    ensure_payload_indexes(client)
    created_once = list(client.created_indexes)
    assert len(created_once) > 0

    ensure_schema(client)
    ensure_payload_indexes(client)
    assert client.created_indexes == created_once
