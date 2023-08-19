from typing import List, Tuple

import numpy as np
import psycopg
from engine.base_client.distances import Distance

from engine.base_client.search import BaseSearcher
from engine.clients.pgvectors.config import PGVECTORS_DB_CONFIG
from engine.clients.pgvectors.parser import PgvectorsConditionParser
from engine.clients.redis.config import REDIS_PORT
from engine.clients.redis.parser import RedisConditionParser


class PgvectorsSearcher(BaseSearcher):
    search_params = {}
    client = None
    parser = PgvectorsConditionParser()

    
    @classmethod
    def get_mp_start_method(cls):
        return 'spawn'

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        config = PGVECTORS_DB_CONFIG
        config['host'] = host
        cls.client =  psycopg.connect(**config)
        cls.search_params = search_params
        cls.distance = distance
        print(search_params)

    @classmethod
    def search_one(cls, vector, meta_conditions, top) -> List[Tuple[int, float]]:        
        conditions = cls.parser.parse(meta_conditions)
        vec_str = str(vector)
        print(meta_conditions)
        print(conditions)
        # exit()
        operator_mapping = {
            Distance.COSINE: "<=>",
            Distance.L2: '<->',
            Distance.DOT: '<#>',
        }
        if conditions is None:
            prefilter_condition = "*"
            params = {}
            query = """
            SELECT id, embedding {operator} %s AS vector_score FROM train ORDER BY embedding {operator} %s LIMIT %s;
            """.format(operator=operator_mapping[cls.distance])
        else:            
            query = """
            SELECT id, embedding {operator} %s AS vector_score FROM train 
            WHERE {conditions}
            ORDER BY embedding {operator} %s LIMIT %s;
            """.format(operator=operator_mapping[cls.distance], conditions=conditions)
        # print("query: ",query)
        with cls.client.cursor() as cursor:
            cursor.execute(
                query, (vec_str, vec_str, top))
            results = cursor.fetchall()
        return results
