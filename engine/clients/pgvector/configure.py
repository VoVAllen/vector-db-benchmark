
from benchmark.dataset import Dataset
from engine.base_client import IncompatibilityError
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
import psycopg

from engine.clients.pgvector.config import PGVECTOR_DB_CONFIG

RECREATE_DDL = """
    DROP TABLE IF EXISTS train;
    CREATE TABLE train (id integer PRIMARY KEY, embedding vector({dims}) NOT NULL {extra_fields});
"""
CLEAN_DDL = """
    DROP TABLE IF EXISTS test;
"""


class PgvectorConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: "l2_ops",
        Distance.COSINE: "cosine_ops",
        Distance.DOT: "dot_ops",
    }
    FIELD_MAPPING = {
        "int": 'integer',
        'float': 'float',
        # "geo": GeoField,
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        config = PGVECTOR_DB_CONFIG
        config['host'] = host

        self.conn = psycopg.connect(**config)
        with self.conn.cursor() as cursor:
            cursor.execute('DROP TABLE IF EXISTS train;')
            cursor.execute('DROP EXTENSION IF EXISTS vector')
            cursor.execute('CREATE EXTENSION vector')
        self.conn.commit()

    def clean(self):
        print("clean")
        with self.conn.cursor() as cursor:
            cursor.execute(CLEAN_DDL)
        self.conn.commit()

    def recreate(self, dataset: Dataset, collection_params):
        print("recreate")
        print(collection_params)
        schema = dataset.config.schema
        for (k, v) in schema.items():
            if v not in self.FIELD_MAPPING:
                raise IncompatibilityError(f"Unsupported field type {v}")
        extra_fields = ', '.join(
            [f'{k} {self.FIELD_MAPPING[v]}' for (k, v) in schema.items()])
        if len(extra_fields) > 0:
            extra_fields = ', ' + extra_fields
        ddl = RECREATE_DDL.format(dims=dataset.config.vector_size, extra_fields=extra_fields)
        print(ddl)
        

        with self.conn.cursor() as cursor:
            cursor.execute(ddl)
        self.conn.commit()

        
        schema = dataset.config.schema
        if len(schema)>0:
            print("schema: ", schema)
            with self.conn.cursor() as cursor:
                for k in schema.keys():
                    cursor.execute(
                        f"CREATE INDEX ON train ({k})")
        else:
            print("no extra schemas")
        self.conn.commit()
