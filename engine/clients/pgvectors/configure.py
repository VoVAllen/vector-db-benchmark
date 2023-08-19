
from benchmark.dataset import Dataset
from engine.base_client import IncompatibilityError
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
import psycopg

from engine.clients.pgvectors.config import PGVECTORS_DB_CONFIG

RECREATE_DDL = """
    DROP TABLE IF EXISTS train;
    CREATE TABLE train (id integer PRIMARY KEY, embedding vector({dims}) NOT NULL);
"""
CLEAN_DDL = """
    DROP TABLE IF EXISTS test;
"""

class PgvectorsConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: "l2_ops",
        Distance.COSINE: "cosine_ops",
        Distance.DOT: "dot_ops",
    }
    INDEX_TYPE_MAPPING = {
        "int": "long",
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        config = PGVECTORS_DB_CONFIG
        config['host'] = host
        
        self.conn = psycopg.connect(**config)
        with self.conn.cursor() as cursor:
            cursor.execute('DROP TABLE IF EXISTS train;')
            cursor.execute('DROP EXTENSION IF EXISTS vectors')
            cursor.execute('CREATE EXTENSION vectors')
        self.conn.commit()

    def clean(self):
        print("clean")
        with self.conn.cursor() as cursor:
            cursor.execute(CLEAN_DDL)
        self.conn.commit()

    def recreate(self, dataset: Dataset, collection_params):
        print("recreate")
        print(RECREATE_DDL.format(dims=dataset.config.vector_size))
        with self.conn.cursor() as cursor:
            cursor.execute(RECREATE_DDL.format(dims=dataset.config.vector_size))
        self.conn.commit()