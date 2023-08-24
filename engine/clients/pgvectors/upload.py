import uuid
from typing import List, Optional
from benchmark.dataset import Dataset

import psycopg
from engine.base_client.distances import Distance


from engine.base_client.upload import BaseUploader
from engine.clients.pgvectors.config import PGVECTORS_DB_CONFIG

INSERT_TRAIN = "INSERT INTO train VALUES (%s, %s)"
CREATE_INDEX = """
    CREATE INDEX ON train USING vectors (embedding {distance_op})
    WITH (options = $$
    capacity = 2097152
    [vectors]
    memmap = "ram"
    [algorithm.hnsw]
    memmap = "ram"
    m = 16
    ef = 128
    $$);
"""


class PgvectorsUploader(BaseUploader):
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        init_params = {
            **{
                "verify_certs": False,
                "request_timeout": 90,
                "retry_on_timeout": True,
            },
            **connection_params,
        }
        config = PGVECTORS_DB_CONFIG
        config['host'] = host
        cls.client = psycopg.connect(**config)
        cls.upload_params = upload_params
        cls.distance = distance
        cls.metadata_key = []

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        if metadata is None:
            operations = []
            INSERT_TRAIN = "INSERT INTO train VALUES (%s, %s)"
            table_train = []
            cls.metadata_key = []
            for idx, vector, payload in zip(ids, vectors):
                table_train.append((idx, str(vector)))
            with cls.client.cursor() as cursor:
                cursor.executemany(
                    INSERT_TRAIN, table_train)
            cls.client.commit()
        else:
            fields_template = ','.join(
                ['%s' for _ in range(len(metadata[0])+2)])
            cls.metadata_key = list(metadata[0].keys())
            INSERT_TRAIN = "INSERT INTO train VALUES ({})".format(
                fields_template)
            table_train = []
            for idx, vector, payload in zip(ids, vectors, metadata):
                fields = [idx, str(vector)]
                for k, v in payload.items():
                    fields.append(v)
                table_train.append(fields)
            with cls.client.cursor() as cursor:
                cursor.executemany(
                    INSERT_TRAIN, table_train)
            cls.client.commit()

    @classmethod
    def post_upload(cls, _distance):
        DISTANCE_MAPPING = {
            Distance.L2: "l2_ops",
            Distance.COSINE: "cosine_ops",
            Distance.DOT: "dot_ops",
        }
        with cls.client.cursor() as cursor:
            cursor.execute(CREATE_INDEX.format(
                distance_op=DISTANCE_MAPPING[_distance]))
        cls.client.commit()

    @classmethod
    def delete_client(cls):
        cls.client.close()
